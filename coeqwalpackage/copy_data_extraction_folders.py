"""Validate scenario spreadsheet ModelFilesLink values against Google Drive folders.

This script reads an Excel scenario listing and validates that each ModelFilesLink
is a Drive folder containing a subfolder named Data_Extraction.

Authentication:
- Best option: service account credentials via GOOGLE_APPLICATION_CREDENTIALS or
  --service-account-file. If access requires acting as a specific user, use
  --delegated-user with a service account that has domain-wide delegation.
- Fallback: OAuth installed-app flow using client secrets provided with
  --oauth-client-secrets-file. This stores a token in the current directory.

Dependencies:
  pip install --upgrade google-api-python-client google-auth google-auth-httplib2 \
      google-auth-oauthlib pandas openpyxl
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from openpyxl import load_workbook
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from google_auth_oauthlib.flow import InstalledAppFlow, WSGITimeoutError
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
import mimetypes

SCOPES = ["https://www.googleapis.com/auth/drive"]
REQUIRED_COLUMNS = [
    "Index",
    "StudyName",
    "GoogleDriveFolderName",
    "ModelFilesLink",
    "HydroClimate",
    "ShortDescription",
    "DV_Path",
    "SV_Path",
    "Start_Date",
    "End_Date",
    "Source",
]
DRIVE_FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"


def parse_drive_folder_id(link: str) -> Optional[str]:
    if not isinstance(link, str) or not link.strip():
        return None

    link = link.strip()
    patterns = [
        r"drive\.google\.com/drive/folders/([a-zA-Z0-9_-]+)",
        r"drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)",
        r"drive\.google\.com/file/d/([a-zA-Z0-9_-]+)",
        r"id=([a-zA-Z0-9_-]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, link)
        if match:
            return match.group(1)

    # If the link is just an ID, accept it only if it is plausibly a Drive ID.
    # This avoids treating simple text labels like 's0107_adjBL_cqlTAI_wTUCP' as folder IDs.
    if re.fullmatch(r"[a-zA-Z0-9_-]+", link) and len(link) >= 25:
        return link

    return None


def get_excel_hyperlink_target(worksheet, column_letter: str, row_index: int) -> Optional[str]:
    try:
        cell = worksheet[f"{column_letter}{row_index + 2}"]
        if cell.hyperlink:
            return str(cell.hyperlink.target).strip()

        if isinstance(cell.value, str) and cell.value.strip().upper().startswith("=HYPERLINK("):
            match = re.search(r'=HYPERLINK\(\s*"([^"\)]+)"', cell.value, re.IGNORECASE)
            if match:
                return match.group(1).strip()
    except Exception:
        return None
    return None


def get_drive_service(
    service_account_file: Optional[Path] = None,
    delegated_user: Optional[str] = None,
    oauth_client_secrets_file: Optional[Path] = None,
    token_file: Optional[Path] = None,
):
    if service_account_file is None:
        env_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        service_account_file = Path(env_path) if env_path else None

    if service_account_file and service_account_file.exists():
        credentials = service_account.Credentials.from_service_account_file(
            service_account_file, scopes=SCOPES
        )
        if delegated_user:
            credentials = credentials.with_subject(delegated_user)
        drive_service = build("drive", "v3", credentials=credentials, cache_discovery=False)
        return drive_service, credentials

    if oauth_client_secrets_file and oauth_client_secrets_file.exists():
        token_path = token_file or Path("token.json")
        creds = None
        if token_path.exists():
            try:
                from google.oauth2.credentials import Credentials

                creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
            except Exception:
                creds = None

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(oauth_client_secrets_file), SCOPES
                )
                try:
                    creds = flow.run_local_server(port=0)
                except WSGITimeoutError:
                    # Local server did not receive the redirect in time (firewall/redirect issues).
                    # Fall back to console-based flow where the user pastes the auth code.
                    print(
                        "Local server auth timed out; falling back to console copy-paste flow."
                    )
                    creds = flow.run_console()

            with open(token_path, "w", encoding="utf-8") as token_handle:
                token_handle.write(creds.to_json())

        drive_service = build("drive", "v3", credentials=creds, cache_discovery=False)
        return drive_service, creds

    raise RuntimeError(
        "No valid authentication configured. Set GOOGLE_APPLICATION_CREDENTIALS or "
        "provide --service-account-file or --oauth-client-secrets-file."
    )


def get_authorized_account_email(credentials) -> Optional[str]:
    """Return the authorized account email for given credentials if available."""
    # Service account
    try:
        service_email = getattr(credentials, "service_account_email", None)
        if service_email:
            return service_email
    except Exception:
        pass

    # OAuth user credentials: query the oauth2 userinfo endpoint
    try:
        oauth2_service = build("oauth2", "v2", credentials=credentials, cache_discovery=False)
        userinfo = oauth2_service.userinfo().get().execute()
        return userinfo.get("email")
    except Exception:
        return None


def validate_spreadsheet_columns(df: pd.DataFrame) -> None:
    read_columns = [str(col).strip() for col in df.columns]
    missing = [col for col in REQUIRED_COLUMNS if col not in read_columns]
    if missing:
        raise ValueError(
            f"Spreadsheet is missing required columns: {', '.join(missing)}"
        )


def folder_contains_data_extraction(drive_service, folder_id: str) -> bool:
    query = (
        f"'{folder_id}' in parents and trashed = false "
        f"and name = 'Data_Extraction' "
        f"and mimeType = '{DRIVE_FOLDER_MIME_TYPE}'"
    )
    response = (
        drive_service.files()
        .list(
            q=query,
            fields="files(id,name,mimeType)",
            pageSize=10,
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
        )
        .execute()
    )
    files = response.get("files") or []
    if not files:
        return None
    return files[0].get("id")


def find_child_folder_by_name(drive_service, parent_id: str, name: str) -> Optional[str]:
    q = (
        f"'{parent_id}' in parents and trashed = false and name = '{name}' "
        f"and mimeType = '{DRIVE_FOLDER_MIME_TYPE}'"
    )
    resp = (
        drive_service.files()
        .list(q=q, fields="files(id,name)", pageSize=10, includeItemsFromAllDrives=True, supportsAllDrives=True)
        .execute()
    )
    files = resp.get("files") or []
    if not files:
        return None
    return files[0].get("id")


def create_drive_folder(drive_service, name: str, parent_id: str) -> str:
    body = {"name": name, "mimeType": DRIVE_FOLDER_MIME_TYPE, "parents": [parent_id]}
    resp = (
        drive_service.files()
        .create(body=body, fields="id,name", supportsAllDrives=True)
        .execute()
    )
    return resp.get("id")


def delete_drive_item(drive_service, item_id: str) -> None:
    try:
        drive_service.files().update(
            fileId=item_id,
            body={"trashed": True},
            supportsAllDrives=True,
        ).execute()
        print(f"Successfully moved {item_id} to the trash.")
    except Exception as exc:
        print(f"An error occurred while trashing {item_id}: {exc}")
        raise


def upload_local_folder_to_drive(drive_service, local_folder: Path, parent_drive_id: str) -> None:
    # Map local dirs to drive folder ids
    folder_map = {str(local_folder): parent_drive_id}
    for root, dirs, files in os.walk(local_folder):
        root_path = Path(root)
        parent_local = str(root_path)
        parent_drive = folder_map.get(parent_local)
        # create child folders
        for d in dirs:
            local_dpath = root_path / d
            drive_folder_id = create_drive_folder(drive_service, d, parent_drive)
            folder_map[str(local_dpath)] = drive_folder_id
        # upload files
        for f in files:
            local_file = root_path / f
            mime_type, _ = mimetypes.guess_type(str(local_file))
            media = MediaFileUpload(str(local_file), mimetype=mime_type or "application/octet-stream", resumable=True)
            metadata = {"name": f, "parents": [parent_drive]}
            drive_service.files().create(body=metadata, media_body=media, fields="id", supportsAllDrives=True).execute()


def validate_drive_folder(drive_service, folder_id: str) -> Dict[str, str]:
    result = {
        "folder_id": folder_id,
        "is_folder": "false",
        "accessible": "false",
        "data_extraction_present": "false",
        "data_extraction_id": "",
        "error": "",
    }
    try:
        metadata = (
            drive_service.files()
            .get(fileId=folder_id, fields="id,name,mimeType", supportsAllDrives=True)
            .execute()
        )
        result["accessible"] = "true"
        if metadata.get("mimeType") == DRIVE_FOLDER_MIME_TYPE:
            result["is_folder"] = "true"
            data_extraction_id = folder_contains_data_extraction(drive_service, folder_id)
            if data_extraction_id:
                result["data_extraction_present"] = "true"
                result["data_extraction_id"] = data_extraction_id
        else:
            result["error"] = "Link does not point to a Drive folder."
    except HttpError as exc:
        # Provide detailed HTTP error information for debugging permission issues
        try:
            status = exc.resp.status
        except Exception:
            status = getattr(exc, "status_code", None)
        content = getattr(exc, "content", None)
        if isinstance(content, bytes):
            try:
                extra = content.decode("utf-8", errors="ignore")
            except Exception:
                extra = str(content)
        else:
            extra = str(content)

        if status == 404:
            result["error"] = f"Folder not found or inaccessible. HTTP {status} {extra}"
        elif status == 403:
            result["error"] = f"Permission denied for this Drive folder. HTTP {status} {extra}"
        else:
            result["error"] = f"Drive API error: HTTP {status} {extra}".strip()
    except Exception as exc:
        result["error"] = f"Unexpected error: {exc}"
    return result


def validate_scenario_listing(
    input_path: Path,
    drive_service,
) -> List[Dict[str, str]]:
    df = pd.read_excel(input_path, engine="openpyxl")
    validate_spreadsheet_columns(df)
    workbook = load_workbook(filename=str(input_path), data_only=True)
    worksheet = workbook.active
    hyperlink_column = None
    for cell in worksheet[1]:
        if cell.value and str(cell.value).strip() == "ModelFilesLink":
            hyperlink_column = cell.column_letter
            break

    results: List[Dict[str, str]] = []

    for row_index, row in df.iterrows():
        visible_link = str(row.get("ModelFilesLink", "")).strip()
        folder_link = visible_link
        folder_id = parse_drive_folder_id(folder_link)

        if hyperlink_column:
            hyperlink_target = get_excel_hyperlink_target(worksheet, hyperlink_column, row_index)
            if hyperlink_target:
                hyperlink_id = parse_drive_folder_id(hyperlink_target)
                if hyperlink_id:
                    folder_link = hyperlink_target
                    folder_id = hyperlink_id

        row_result = {
            "row": row_index + 1,
            "Index": str(row.get("Index", "")),
            "StudyName": str(row.get("StudyName", "")),
            "ModelFilesLink": visible_link,
            "ResolvedModelFilesLink": folder_link if folder_link != visible_link else "",
            "DV_Path": str(row.get("DV_Path", "")),
            "folder_id": folder_id or "",
            "is_valid_drive_folder": "false",
            "has_data_extraction": "false",
            "message": "",
        }

        if not folder_link:
            row_result["message"] = "ModelFilesLink is empty."
            results.append(row_result)
            continue

        if not folder_id:
            row_result["message"] = "Could not parse a Google Drive folder ID from ModelFilesLink."
            results.append(row_result)
            continue

        validation = validate_drive_folder(drive_service, folder_id)
        row_result["folder_id"] = validation["folder_id"]
        if validation["accessible"] == "true" and validation["is_folder"] == "true":
            row_result["is_valid_drive_folder"] = "true"
            if validation["data_extraction_present"] == "true":
                row_result["has_data_extraction"] = "true"
                row_result["data_extraction_id"] = validation.get("data_extraction_id", "")
                row_result["message"] = "Valid folder and Data_Extraction subfolder found."
            else:
                row_result["message"] = "Folder valid but Data_Extraction subfolder not found."
        else:
            row_result["message"] = validation["error"] or "Invalid folder link."

        results.append(row_result)

    return results


def print_results(results: List[Dict[str, str]]) -> None:
    for result in results:
        print("---")
        print(f"Row: {result['row']}  Index: {result['Index']}  StudyName: {result['StudyName']}")
        print(f"ModelFilesLink: {result['ModelFilesLink']}")
        print(f"Drive Folder ID: {result['folder_id']}")
        print(f"Valid Drive Folder: {result['is_valid_drive_folder']}")
        print(f"Data_Extraction Present: {result['has_data_extraction']}")
        print(f"Message: {result['message']}")
    print("---")
    print(f"Validated {len(results)} rows.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate scenario listing Google Drive links and Data_Extraction subfolders."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path('.'),
        help=(
            "Base directory containing the scenario xlsx and local folders. "
            "The script will search this directory (and common subfolders) for the scenario spreadsheet."
        ),
    )
    parser.add_argument(
        "--run-type",
        choices=["COPY", "CLEAN"],
        default="COPY",
        help="Operation mode: COPY to push local Data_Extraction to Drive; CLEAN to remove Data_Extraction_BU backups on Drive.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview actions without changing Google Drive contents.",
    )
    parser.add_argument(
        "--service-account-file",
        type=Path,
        help="Path to a Google service account JSON credentials file.",
    )
    parser.add_argument(
        "--delegated-user",
        help="Optional user email to impersonate when using a service account with domain-wide delegation.",
    )
    parser.add_argument(
        "--oauth-client-secrets-file",
        type=Path,
        help="Path to OAuth 2.0 client secrets JSON file for user consent flow.",
    )
    parser.add_argument(
        "--token-file",
        type=Path,
        default=Path("token.json"),
        help="Path to store OAuth user credentials token.",
    )
    args = parser.parse_args()

    # Resolve the input spreadsheet by searching the provided base directory.
    base_dir = args.base_dir or Path(".")
    if not base_dir.exists() or not base_dir.is_dir():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    default_names = [
        "coeqwal_cs3_scenario_listing_v7.xlsx",
        "coeqwal_cs3_scenario_listing_v6.xlsx",
        "coeqwal_cs3_scenario_listing_v5.xlsx",
    ]

    input_path: Optional[Path] = None
    last_err = None
    try:
        # check for common filenames in the base dir
        for name in default_names:
            cand = base_dir / name
            if cand.exists() and cand.is_file():
                input_path = cand
                break

        # pick the first spreadsheet in the base dir
        if input_path is None:
            for f in base_dir.iterdir():
                if f.is_file() and f.suffix.lower() in (".xlsx", ".xls"):
                    input_path = f
                    break

        # also try common subfolders
        if input_path is None:
            subfolders = [base_dir / "Scenarios", base_dir / "CalSim3_Model_Runs_BAK" / "Scenarios"]
            for sp in subfolders:
                try:
                    if sp.exists() and sp.is_dir():
                        for f in sp.iterdir():
                            if f.is_file() and f.suffix.lower() in (".xlsx", ".xls"):
                                input_path = f
                                break
                except PermissionError as pe:
                    last_err = pe
                if input_path:
                    break

        if input_path is None:
            if last_err:
                raise PermissionError(f"Permission denied accessing '{base_dir}': {last_err}")
            raise FileNotFoundError(f"No spreadsheet (*.xls, *.xlsx) found in base-dir: {base_dir}")
    except PermissionError as pe:
        raise PermissionError(f"Permission denied accessing '{base_dir}': {pe}")

    try:
        if not input_path.exists():
            raise FileNotFoundError(f"Input spreadsheet not found: {input_path}")
        # If input_path is a directory (user passed a folder), try to find the spreadsheet inside it
        if input_path.is_dir():
            try:
                v7 = input_path / "coeqwal_cs3_scenario_listing_v7.xlsx"
                v7_xls = input_path / "coeqwal_cs3_scenario_listing_v7.xls"
                found = None
                for cand in (v7, v7_xls):
                    if cand.exists() and cand.is_file():
                        found = cand
                        break

                if not found:
                    # pick the first xlsx/xls file in the directory
                    for f in input_path.iterdir():
                        if f.suffix.lower() in (".xlsx", ".xls") and f.is_file():
                            found = f
                            break

                if not found:
                    raise FileNotFoundError(f"No spreadsheet (*.xls, *.xlsx) found in directory: {input_path}")

                input_path = found
            except PermissionError as pe:
                raise PermissionError(f"Permission denied accessing '{input_path}': {pe}")
    except PermissionError as pe:
        raise PermissionError(f"Permission denied accessing '{input_path}': {pe}")

    drive_service, credentials = get_drive_service(
        service_account_file=args.service_account_file,
        delegated_user=args.delegated_user,
        oauth_client_secrets_file=args.oauth_client_secrets_file,
        token_file=args.token_file,
    )

    acct = get_authorized_account_email(credentials)
    if acct:
        print(f"Authorized as: {acct}")
    else:
        print("Authorized account: (unknown)")

    # Attempt to get Drive 'about' info to show which Drive user the API sees
    try:
        about = drive_service.about().get(fields="user").execute()
        user = about.get("user", {})
        if user:
            print(f"Drive API user: {user.get('emailAddress')} ({user.get('displayName')})")
    except Exception as e:
        print(f"Could not get Drive about user: {e}")

    print(f"Run mode: {args.run_type}{' (dry-run)' if args.dry_run else ''}")
    results = validate_scenario_listing(input_path, drive_service)

    # Post-processing: check local dirs and perform COPY/CLEAN actions
    base_dir = args.base_dir or Path(".")
    summary = {"total": 0, "errors": [], "creates": 0, "updates": 0, "cleans": 0}

    for row in results:
        summary["total"] += 1
        study = row.get("StudyName") or f"row_{row.get('row') }"
        if args.run_type == "CLEAN":
            op_type = "CLEAN"
        else:
            op_type = "UPDATE" if row.get("has_data_extraction") == "true" else "CREATE"
        print(f"[{summary['total']}] Processing {study} ({op_type})...")

        # Drive folder existence error
        if row.get("is_valid_drive_folder") != "true":
            reason = row.get("message")
            print(f"  ⚠️  SKIP: {reason}")
            summary["errors"].append({"study": study, "reason": reason})
            continue

        parent_drive_id = row.get("folder_id")
        if args.run_type == "CLEAN":
            # CLEAN only needs the Drive folder and Data_Extraction_BU existence.
            try:
                if args.dry_run:
                    print(f"  [DRY-RUN] Would remove Data_Extraction_BU from Drive folder {parent_drive_id} if present")
                else:
                    print(f"  ↳ CLEAN: removing Data_Extraction_BU backup if present...")
                    bu_id = find_child_folder_by_name(drive_service, parent_drive_id, "Data_Extraction_BU")
                    if bu_id:
                        delete_drive_item(drive_service, bu_id)
                        print(f"    ✓ deleted Data_Extraction_BU")
                    else:
                        print(f"    ℹ no Data_Extraction_BU found (nothing to clean)")
            except Exception as exc:
                print(f"  ✗ ERROR: {exc}")
                summary["errors"].append({"study": study, "reason": f"Run-time error: {exc}"})
                continue

            print(f"  ✓ {study} complete\n")
            summary.setdefault("cleans", 0)
            summary["cleans"] += 1
            continue

        is_update = row.get("has_data_extraction") == "true"
        if is_update:
            summary["updates"] += 1
        else:
            summary["creates"] += 1

        # Local DV_Path first token
        dv_raw = row.get("DV_Path", "")
        dv_path = str(dv_raw).strip() if dv_raw is not None else ""
        if dv_path.lower() in {"nan", "none"}:
            dv_path = ""

        first_token = None
        normalized = ""
        tokens: List[str] = []
        if dv_path:
            normalized = dv_path.replace("/", "\\")
            tokens = [t for t in re.split(r"[\\/]+", normalized) if t]
            if tokens:
                first_token = tokens[0]

        if not first_token:
            print(
                f"  ⚠️  SKIP: DV_Path missing or malformed (raw={repr(dv_raw)}, normalized={repr(normalized)})"
            )
            summary["errors"].append({"study": study, "reason": "DV_Path missing or malformed"})
            continue

        local_dir = (base_dir / first_token).resolve()
        print(f"  → Local directory: {local_dir}")
        if not local_dir.exists() or not local_dir.is_dir():
            reason = f"Local dir missing: {local_dir}"
            print(f"  ⚠️  SKIP: {reason}")
            summary["errors"].append({"study": study, "reason": reason})
            continue

        local_data_extraction = local_dir / "Data_Extraction"
        if not local_data_extraction.exists() or not local_data_extraction.is_dir():
            reason = f"Local Data_Extraction missing in {local_dir}"
            print(f"  ⚠️  SKIP: {reason}")
            summary["errors"].append({"study": study, "reason": reason})
            continue

        print(f"  → Data_Extraction folder: {local_data_extraction}")

        # Perform run-type actions
        try:
            if args.run_type == "COPY":
                if args.dry_run:
                    print(f"  [DRY-RUN] Would copy '{local_data_extraction}' to Drive folder {parent_drive_id}")
                    if row.get("data_extraction_id"):
                        print(f"  [DRY-RUN] Would rename existing Data_Extraction → Data_Extraction_BU")
                else:
                    existing_de_id = row.get("data_extraction_id")
                    if existing_de_id:
                        print(f"  ↳ UPDATE: renaming existing Data_Extraction to Data_Extraction_BU...")
                        # check for existing BU
                        bu_id = find_child_folder_by_name(drive_service, parent_drive_id, "Data_Extraction_BU")
                        if bu_id:
                            delete_drive_item(drive_service, bu_id)
                            print(f"    ✓ deleted old Data_Extraction_BU")
                        # rename existing
                        drive_service.files().update(fileId=existing_de_id, body={"name": "Data_Extraction_BU"}, supportsAllDrives=True).execute()
                        print(f"    ✓ renamed to Data_Extraction_BU")
                    else:
                        print(f"  ↳ CREATE: new Data_Extraction folder on Drive...")
                    # create new Data_Extraction folder
                    new_de_id = create_drive_folder(drive_service, "Data_Extraction", parent_drive_id)
                    print(f"    ✓ created Data_Extraction folder")
                    # upload local contents
                    print(f"  ↳ uploading local folder contents ({local_data_extraction})...")
                    upload_local_folder_to_drive(drive_service, local_data_extraction, new_de_id)
                    print(f"    ✓ upload complete")
        except Exception as exc:
            print(f"  ✗ ERROR: {exc}")
            summary["errors"].append({"study": study, "reason": f"Run-time error: {exc}"})
            continue

        print(f"  ✓ {study} complete\n")

    # Print final summary
    print("\n=== Summary ===")
    print(f"Total rows processed: {summary['total']}")
    print(f"Creates: {summary['creates']}  Updates: {summary['updates']}  Cleans: {summary['cleans']}")
    print(f"Errors: {len(summary['errors'])}")
    if summary["errors"]:
        print("Details:")
        for e in summary["errors"]:
            print(f" - {e['study']}: {e['reason']}")


if __name__ == "__main__":
    main()
