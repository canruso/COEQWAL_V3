"""Tests for repo restructure: directory layout, file locations, import chains, and path references.

These are structural validation tests, not unit tests of package logic.
They verify that the restructure moved files correctly, updated all path
references in notebooks, fixed bare imports in package modules, and cleaned
up obsolete directories/files.
"""
from __future__ import annotations

import json
import os
import re
import sys

import pytest

# ---------------------------------------------------------------------------
# Resolve repo root from the test file's location (tests/ is one level deep)
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Ensure repo root is on sys.path so package imports can be tested
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_notebook(name: str) -> dict:
    """Load a notebook JSON from notebooks/ by filename."""
    path = os.path.join(REPO_ROOT, "notebooks", name)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _cell_source(cell: dict) -> str:
    """Join a cell's source list into a single string."""
    src = cell.get("source", [])
    if isinstance(src, list):
        return "".join(src)
    return str(src)


def _code_cells(nb: dict) -> list[dict]:
    """Return only code cells from a notebook."""
    return [c for c in nb["cells"] if c.get("cell_type") == "code"]


def _all_code_source(nb: dict) -> str:
    """Concatenate all code cell sources into one string for searching."""
    return "\n".join(_cell_source(c) for c in _code_cells(nb))


# ---------------------------------------------------------------------------
# 1. Directory structure exists
# ---------------------------------------------------------------------------

class TestDirectoryStructure:

    def test_target_directories_exist(self):
        """All target directories created by the restructure must exist."""
        expected_dirs = [
            "coeqwalpackage",
            "config",
            "data/scenarios",
            "data/variables",
            "data/mappings",
            "data/tiers",
            "data/shapefiles",
            "tests",
            "outputs",
        ]
        for d in expected_dirs:
            full = os.path.join(REPO_ROOT, d)
            assert os.path.isdir(full), f"Directory missing: {d}"


# ---------------------------------------------------------------------------
# 2. Package files exist at new location
# ---------------------------------------------------------------------------

class TestPackageFiles:

    EXPECTED_FILES = [
        "__init__.py",
        "metrics.py",
        "plotting.py",
        "tier.py",
        "cqwlutils.py",
        "llm_utils.py",
        "prompts.py",
        "scenario_review.py",
        "DataExtraction.py",
        "AuxFunctions.py",
        "csPlots.py",
        "cs3.py",
        "cs_util.py",
        "dss3_functions_reference.py",
    ]

    def test_package_files_exist(self):
        """All Python module files exist in top-level coeqwalpackage/."""
        pkg_dir = os.path.join(REPO_ROOT, "coeqwalpackage")
        for fname in self.EXPECTED_FILES:
            full = os.path.join(pkg_dir, fname)
            assert os.path.isfile(full), f"Package file missing: coeqwalpackage/{fname}"


# ---------------------------------------------------------------------------
# 3. Old package directory removed
# ---------------------------------------------------------------------------

class TestOldPackageRemoved:

    def test_old_package_dir_removed(self):
        """notebooks/coeqwalpackage/ should not exist after restructure."""
        old_dir = os.path.join(REPO_ROOT, "notebooks", "coeqwalpackage")
        assert not os.path.exists(old_dir), (
            "Old package directory still exists: notebooks/coeqwalpackage/"
        )


# ---------------------------------------------------------------------------
# 4. Config files exist
# ---------------------------------------------------------------------------

class TestConfigFiles:

    def test_config_files_exist(self):
        """Excel init files and wresl file are in config/."""
        expected = [
            "CalSim3DataExtractionInitFile_v4.xlsx",
            "CalSim3DeliveryDataExtractionInitFile_v1.xlsx",
            "CalSim3GroundWaterDataExtractionInitFile_v1.xlsx",
            "CalSim3GWregionIndex.wresl",
        ]
        config_dir = os.path.join(REPO_ROOT, "config")
        for fname in expected:
            full = os.path.join(config_dir, fname)
            assert os.path.isfile(full), f"Config file missing: config/{fname}"


# ---------------------------------------------------------------------------
# 5. Data CSVs exist in correct subdirectories
# ---------------------------------------------------------------------------

class TestDataCSVs:

    DATA_FILES = {
        "data/scenarios/coeqwal_cs3_scenario_listing_v5.csv": None,
        "data/scenarios/scenario_groupings.csv": None,
        "data/variables/variable_groupings.csv": None,
        "data/mappings/Agricultural_Mapping.csv": None,
        "data/mappings/DrinkingWater_Mapping.csv": None,
        "data/mappings/Eflows_Mapping.csv": None,
        "data/mappings/CalSim3_WBA.csv": None,
        "data/mappings/sr_to_wba_mapping.csv": None,
        "data/mappings/groundwater_wresl_mapping.csv": None,
        "data/tiers/delta_export_salinity_tiers.csv": None,
        "data/tiers/in_delta_salinity_tiers.csv": None,
        "data/tiers/20250908draft_C2VSim_73-15_WBA_Storage.csv": None,
    }

    def test_data_csvs_exist(self):
        """All CSV files exist in their target data/ subdirectories."""
        for relpath in self.DATA_FILES:
            full = os.path.join(REPO_ROOT, relpath)
            assert os.path.isfile(full), f"Data CSV missing: {relpath}"


# ---------------------------------------------------------------------------
# 6. Shapefiles exist
# ---------------------------------------------------------------------------

class TestShapefiles:

    def test_shapefiles_exist(self):
        """data/shapefiles/ directory contains .shp files."""
        shp_dir = os.path.join(REPO_ROOT, "data", "shapefiles")
        assert os.path.isdir(shp_dir), "data/shapefiles/ directory missing"
        shp_files = [f for f in os.listdir(shp_dir) if f.endswith(".shp")]
        assert len(shp_files) > 0, "No .shp files found in data/shapefiles/"


# ---------------------------------------------------------------------------
# 7. CSVs copied from water-data-dashboard
# ---------------------------------------------------------------------------

class TestDashboardCSVsCopied:

    def test_dashboard_csvs_copied(self):
        """CSVs copied from water-data-dashboard exist in data/."""
        expected = [
            "data/variables/trend_report_variables_v5.csv",
            "data/variables/parallel_variable_groupings.csv",
            "data/tiers/tier_df.csv",
        ]
        for relpath in expected:
            full = os.path.join(REPO_ROOT, relpath)
            assert os.path.isfile(full), (
                f"Dashboard CSV not copied: {relpath}"
            )


# ---------------------------------------------------------------------------
# 8. Package imports work
# ---------------------------------------------------------------------------

class TestPackageImports:

    def test_package_imports(self):
        """Core package modules can be imported without error."""
        # These imports exercise the import chain.
        # If bare imports remain broken, this will fail with ImportError.
        import coeqwalpackage  # noqa: F401
        from coeqwalpackage import metrics  # noqa: F401
        from coeqwalpackage import cqwlutils  # noqa: F401
        from coeqwalpackage.metrics import create_subset_unit  # noqa: F401
        from coeqwalpackage.cqwlutils import read_init_file  # noqa: F401


# ---------------------------------------------------------------------------
# 9. No bare imports in core modules
# ---------------------------------------------------------------------------

class TestNoBareImports:

    # Patterns that indicate bare (unqualified) imports which would break
    # after moving the package to repo root.
    BARE_IMPORT_PATTERNS = [
        (r"^from metrics import", "plotting.py or tier.py"),
        (r"^import cqwlutils\b", "plotting.py or tier.py"),
        (r"^import plotting\b", "tier.py"),
        (r"^from cqwlutils import", "tier.py"),
    ]

    def test_no_bare_imports_in_core_modules(self):
        """plotting.py and tier.py must not have bare (unqualified) imports."""
        for module_name in ("plotting.py", "tier.py"):
            path = os.path.join(REPO_ROOT, "coeqwalpackage", module_name)
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for line_no, line in enumerate(lines, 1):
                stripped = line.strip()
                for pattern, desc in self.BARE_IMPORT_PATTERNS:
                    if re.match(pattern, stripped):
                        pytest.fail(
                            f"Bare import in coeqwalpackage/{module_name} "
                            f"line {line_no}: {stripped!r}"
                        )


# ---------------------------------------------------------------------------
# 10. Notebook sys.path references updated
# ---------------------------------------------------------------------------

class TestNotebookSysPath:

    # All notebooks that previously had sys.path.append('./coeqwalpackage')
    # Scenario_Review.ipynb is excluded - it never had sys.path.append
    NOTEBOOKS_WITH_SYSPATH = [
        "ConvertUnits.ipynb",
        "ConvertDeliveryUnits.ipynb",
        "Metrics.ipynb",
        "Plotting.ipynb",
        "Dashboard.ipynb",
        "Scenario_Review_LLM.ipynb",
        "Tier_Assignment_FloodRisk.ipynb",
        "Tier_Assignment_Salinity.ipynb",
        "Tier_Assignment_Storage.ipynb",
        "Tier_Assignment_GroundWater.ipynb",
        "MergedMetrics.ipynb",
        "ExtractStudiesFromDssAndCompoundVariablesCalSim3.ipynb",
        "ExtractDeliveryVariablesCalSim3.ipynb",
        "ExtractGroundWaterDataCalSim3.ipynb",
        "ComputeGroundWaterTrends.ipynb",
        "ParallelPlots.ipynb",
    ]

    def test_notebook_syspath_updated(self):
        """All notebooks use sys.path.insert(0, '..') not './coeqwalpackage'."""
        for nb_name in self.NOTEBOOKS_WITH_SYSPATH:
            nb = _load_notebook(nb_name)
            all_src = _all_code_source(nb)

            assert "./coeqwalpackage" not in all_src, (
                f"{nb_name} still contains './coeqwalpackage' in sys.path"
            )
            assert "sys.path.insert(0, '..')" in all_src or "sys.path.insert(0, \"..\")" in all_src, (
                f"{nb_name} missing sys.path.insert(0, '..')"
            )


# ---------------------------------------------------------------------------
# 11. Notebook CtrlFile references point to ../config/
# ---------------------------------------------------------------------------

class TestNotebookCtrlFilePaths:

    # Map of notebook -> expected CtrlFile value after restructure
    NOTEBOOKS_CTRLFILE = {
        "ConvertUnits.ipynb": "../config/CalSim3DataExtractionInitFile_v4.xlsx",
        "ConvertDeliveryUnits.ipynb": "../config/CalSim3DeliveryDataExtractionInitFile_v1.xlsx",
        "Metrics.ipynb": "../config/CalSim3DataExtractionInitFile_v4.xlsx",
        "Plotting.ipynb": "../config/CalSim3DataExtractionInitFile_v4.xlsx",
        "Dashboard.ipynb": "../config/CalSim3DataExtractionInitFile_v4.xlsx",
        "Scenario_Review_LLM.ipynb": "../config/CalSim3DataExtractionInitFile_v4.xlsx",
        "Tier_Assignment_FloodRisk.ipynb": "../config/CalSim3DataExtractionInitFile_v4.xlsx",
        "Tier_Assignment_Salinity.ipynb": "../config/CalSim3DataExtractionInitFile_v4.xlsx",
        "Tier_Assignment_Storage.ipynb": "../config/CalSim3DataExtractionInitFile_v4.xlsx",
        "MergedMetrics.ipynb": "../config/CalSim3DataExtractionInitFile_v4.xlsx",
        "ExtractStudiesFromDssAndCompoundVariablesCalSim3.ipynb": "../config/CalSim3DataExtractionInitFile_v4.xlsx",
        "ExtractDeliveryVariablesCalSim3.ipynb": "../config/CalSim3DeliveryDataExtractionInitFile_v1.xlsx",
        "ExtractGroundWaterDataCalSim3.ipynb": "../config/CalSim3GroundWaterDataExtractionInitFile_v1.xlsx",
        "Tier_Assignment_GroundWater.ipynb": "../config/CalSim3GroundWaterDataExtractionInitFile_v1.xlsx",
        "ComputeGroundWaterTrends.ipynb": "../config/CalSim3GroundWaterDataExtractionInitFile_v1.xlsx",
    }

    def test_notebook_ctrlfile_paths(self):
        """All notebooks reference CtrlFile with '../config/' prefix."""
        for nb_name, expected_path in self.NOTEBOOKS_CTRLFILE.items():
            nb = _load_notebook(nb_name)
            all_src = _all_code_source(nb)

            # The CtrlFile variable name varies: CtrlFile or CTRL_FILE
            # Look for the expected path string anywhere in code
            assert expected_path in all_src, (
                f"{nb_name}: expected CtrlFile path '{expected_path}' not found. "
                f"Old path may still be present."
            )

            # Also verify the OLD path (without ../config/) is gone
            basename = os.path.basename(expected_path)
            # Match bare filename assignment like CtrlFile = 'CalSim3..xlsx'
            # but NOT when it's inside ../config/
            old_patterns = [
                f"= '{basename}'",
                f'= "{basename}"',
            ]
            for pat in old_patterns:
                if pat in all_src:
                    pytest.fail(
                        f"{nb_name}: still has bare CtrlFile path: {pat}"
                    )


# ---------------------------------------------------------------------------
# 12. Notebook CSV paths - MergedMetrics
# ---------------------------------------------------------------------------

class TestMergedMetricsCSVPaths:

    def test_notebook_csv_paths_merged_metrics(self):
        """MergedMetrics notebook CSV reads point to ../data/ paths."""
        nb = _load_notebook("MergedMetrics.ipynb")
        all_src = _all_code_source(nb)

        # These are the CSV reads that must be updated
        expected_paths = [
            "../data/scenarios/scenario_groupings.csv",
            "../data/mappings/Agricultural_Mapping.csv",
            "../data/mappings/DrinkingWater_Mapping.csv",
            "../data/mappings/Eflows_Mapping.csv",
        ]
        for path in expected_paths:
            assert path in all_src, (
                f"MergedMetrics.ipynb: expected CSV path '{path}' not found"
            )

        # Old bare CSV refs should be gone
        bare_csvs = [
            "read_csv('scenario_groupings.csv')",
            "read_csv('Agricultural_Mapping.csv')",
            "read_csv('DrinkingWater_Mapping.csv')",
            "read_csv('Eflows_Mapping.csv')",
        ]
        for bare in bare_csvs:
            assert bare not in all_src, (
                f"MergedMetrics.ipynb: still has bare CSV path: {bare}"
            )


# ---------------------------------------------------------------------------
# 13. Notebook CSV paths - Dashboard
# ---------------------------------------------------------------------------

class TestDashboardCSVPaths:

    def test_notebook_csv_paths_dashboard(self):
        """Dashboard notebook data paths point to ../data/ paths."""
        nb = _load_notebook("Dashboard.ipynb")
        all_src = _all_code_source(nb)

        # The Dashboard cell[16] standalone Dash app has these paths
        expected_paths = [
            "../data/variables/trend_report_variables_v5.csv",
            "../data/scenarios/coeqwal_cs3_scenario_listing_v5.csv",
            "../data/variables/variable_groupings.csv",
            "../data/scenarios/scenario_groupings.csv",
        ]
        for path in expected_paths:
            assert path in all_src, (
                f"Dashboard.ipynb: expected data path '{path}' not found"
            )

        # Old data/ prefixed paths (from dashboard app) should be gone
        old_paths = [
            '"data/trend_report_variables_v5.csv"',
            '"data/coeqwal_cs3_scenario_listing_v5.csv"',
            '"data/variable_groupings.csv"',
            '"data/scenario_groupings.csv"',
        ]
        for old in old_paths:
            assert old not in all_src, (
                f"Dashboard.ipynb: still has old data path: {old}"
            )


# ---------------------------------------------------------------------------
# 14. Archive directories deleted
# ---------------------------------------------------------------------------

class TestArchivesDeleted:

    def test_archives_deleted(self):
        """archive_pre_cleanup dirs at root and notebooks/ are gone."""
        root_archive = os.path.join(REPO_ROOT, "archive_pre_cleanup")
        nb_archive = os.path.join(REPO_ROOT, "notebooks", "archive_pre_cleanup")
        assert not os.path.exists(root_archive), (
            "Root archive_pre_cleanup/ still exists"
        )
        assert not os.path.exists(nb_archive), (
            "notebooks/archive_pre_cleanup/ still exists"
        )


# ---------------------------------------------------------------------------
# 15. No __pycache__ in tracked areas
# ---------------------------------------------------------------------------

class TestNoPycache:

    def test_no_pycache_in_git(self):
        """__pycache__ is gitignored (pytest creates it during runs, so check git status)."""
        gitignore_path = os.path.join(REPO_ROOT, ".gitignore")
        assert os.path.isfile(gitignore_path), ".gitignore missing"
        with open(gitignore_path) as f:
            content = f.read()
        assert "__pycache__" in content, ".gitignore does not include __pycache__"
        assert "*.pyc" in content, ".gitignore does not include *.pyc"


# ---------------------------------------------------------------------------
# 16. .gitignore exists with required rules
# ---------------------------------------------------------------------------

class TestGitignore:

    REQUIRED_PATTERNS = [
        "__pycache__",
        "*.pyc",
        ".DS_Store",
    ]

    def test_gitignore_exists(self):
        """Verify .gitignore exists and has __pycache__, .pyc, .DS_Store rules."""
        gitignore_path = os.path.join(REPO_ROOT, ".gitignore")
        assert os.path.isfile(gitignore_path), ".gitignore missing at repo root"

        with open(gitignore_path, "r", encoding="utf-8") as f:
            content = f.read()

        for pattern in self.REQUIRED_PATTERNS:
            assert pattern in content, (
                f".gitignore missing required pattern: {pattern}"
            )


# ---------------------------------------------------------------------------
# 17. Old CSV/xlsx/wresl files removed from notebooks/
# ---------------------------------------------------------------------------

class TestOldFilesRemoved:

    # Files that should have been moved OUT of notebooks/
    OLD_FILES = [
        "CalSim3DataExtractionInitFile_v4.xlsx",
        "CalSim3DeliveryDataExtractionInitFile_v1.xlsx",
        "CalSim3GroundWaterDataExtractionInitFile_v1.xlsx",
        "CalSim3GWregionIndex.wresl",
        "coeqwal_cs3_scenario_listing_v5.csv",
        "scenario_groupings.csv",
        "variable_groupings.csv",
        "Agricultural_Mapping.csv",
        "DrinkingWater_Mapping.csv",
        "Eflows_Mapping.csv",
        "CalSim3_WBA.csv",
        "sr_to_wba_mapping.csv",
        "groundwater_wresl_mapping.csv",
        "delta_export_salinity_tiers.csv",
        "in_delta_salinity_tiers.csv",
        "20250908draft_C2VSim_73-15_WBA_Storage.csv",
    ]

    def test_old_csv_files_removed(self):
        """CSV/xlsx/wresl files no longer exist in notebooks/ root."""
        nb_dir = os.path.join(REPO_ROOT, "notebooks")
        for fname in self.OLD_FILES:
            full = os.path.join(nb_dir, fname)
            assert not os.path.exists(full), (
                f"Old file still in notebooks/: {fname}"
            )


# ---------------------------------------------------------------------------
# 18. Output directories merged
# ---------------------------------------------------------------------------

class TestOutputDirsMerged:

    def test_output_dirs_merged(self):
        """output/ dir gone, outputs/ exists."""
        old_output = os.path.join(REPO_ROOT, "output")
        new_outputs = os.path.join(REPO_ROOT, "outputs")

        assert not os.path.exists(old_output), (
            "Old output/ directory still exists (should be merged into outputs/)"
        )
        assert os.path.isdir(new_outputs), "outputs/ directory missing"


# ---------------------------------------------------------------------------
# 19. Test files at new location
# ---------------------------------------------------------------------------

class TestTestFilesExist:

    def test_test_files_exist(self):
        """Test files exist in tests/ directory."""
        expected = [
            "test_structured_validation.py",
            "test_compute_var_stats.py",
            "test_export_qv.py",
        ]
        tests_dir = os.path.join(REPO_ROOT, "tests")
        for fname in expected:
            full = os.path.join(tests_dir, fname)
            assert os.path.isfile(full), f"Test file missing: tests/{fname}"


# ---------------------------------------------------------------------------
# 20. Groundwater notebook path references
# ---------------------------------------------------------------------------

class TestGroundwaterNotebookPaths:

    def test_tier_assignment_groundwater_paths(self):
        """Tier_Assignment_GroundWater references data files at ../data/ paths."""
        nb = _load_notebook("Tier_Assignment_GroundWater.ipynb")
        all_src = _all_code_source(nb)

        # These file references must be updated to ../data/ paths
        expected_fragments = [
            "../data/mappings/CalSim3_WBA.csv",
            "../data/tiers/20250908draft_C2VSim_73-15_WBA_Storage.csv",
            "../data/mappings/groundwater_wresl_mapping.csv",
            "../data/shapefiles/",
        ]
        for frag in expected_fragments:
            assert frag in all_src, (
                f"Tier_Assignment_GroundWater.ipynb: expected path '{frag}' not found"
            )

        # Old bare references should be gone
        old_refs = [
            '"CalSim3_WBA.csv"',
            '"groundwater_wresl_mapping.csv"',
            '"./shapefiles/',
        ]
        for old in old_refs:
            assert old not in all_src, (
                f"Tier_Assignment_GroundWater.ipynb: still has old path: {old}"
            )

    def test_compute_groundwater_trends_paths(self):
        """ComputeGroundWaterTrends references config/data files at correct paths."""
        nb = _load_notebook("ComputeGroundWaterTrends.ipynb")
        all_src = _all_code_source(nb)

        # Accept either '../config/file' literal or os.path.join(base_dir, "config", "file") pattern
        wresl_ok = (
            "../config/CalSim3GWregionIndex.wresl" in all_src
            or ('"config"' in all_src and '"CalSim3GWregionIndex.wresl"' in all_src)
        )
        assert wresl_ok, "ComputeGroundWaterTrends.ipynb: wresl not referenced via config/ path"

        wba_ok = (
            "../data/mappings/CalSim3_WBA.csv" in all_src
            or ('"mappings"' in all_src and '"CalSim3_WBA.csv"' in all_src)
        )
        assert wba_ok, "ComputeGroundWaterTrends.ipynb: CalSim3_WBA.csv not referenced via data/mappings/ path"

        # Old bare references should be gone (no direct filename without config/ or data/ path)
        assert 'os.path.join(base_dir, "CalSim3GWregionIndex.wresl")' not in all_src, (
            "ComputeGroundWaterTrends: wresl file referenced without config/ subdirectory"
        )


# ---------------------------------------------------------------------------
# 21. Scenario_Review notebook has sys.path.insert
# ---------------------------------------------------------------------------

class TestScenarioReviewNotebook:

    def test_scenario_review_has_syspath(self):
        """Scenario_Review.ipynb must have sys.path.insert(0, '..') added."""
        nb = _load_notebook("Scenario_Review.ipynb")
        all_src = _all_code_source(nb)

        # The original notebook did NOT have sys.path.append. After restructure,
        # it needs sys.path.insert(0, '..') for the import to work.
        assert "sys.path.insert(0, '..')" in all_src or 'sys.path.insert(0, "..")' in all_src, (
            "Scenario_Review.ipynb: missing sys.path.insert(0, '..') - "
            "the import from coeqwalpackage will fail without it"
        )


# ---------------------------------------------------------------------------
# 22. scenario_review.py self-import removed
# ---------------------------------------------------------------------------

class TestScenarioReviewSelfImport:

    def test_scenario_review_no_self_import(self):
        """scenario_review.py must not contain self-import."""
        path = os.path.join(REPO_ROOT, "coeqwalpackage", "scenario_review.py")
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        # The original had: from coeqwalpackage import scenario_review
        # This is a self-import that causes ImportError. It must be removed.
        assert "from coeqwalpackage import scenario_review" not in content, (
            "coeqwalpackage/scenario_review.py still has self-import"
        )


# ---------------------------------------------------------------------------
# 23. Tier_Assignment_Salinity double CtrlFile
# ---------------------------------------------------------------------------

class TestTierSalinityDoubleCtrlFile:

    def test_tier_salinity_both_ctrlfile_refs_updated(self):
        """Tier_Assignment_Salinity has CtrlFile in two cells - both must be updated."""
        nb = _load_notebook("Tier_Assignment_Salinity.ipynb")
        code_cells = _code_cells(nb)

        # Count how many cells contain the CtrlFile assignment with ../config/
        cells_with_new_path = 0
        cells_with_old_path = 0
        for cell in code_cells:
            src = _cell_source(cell)
            if "CalSim3DataExtractionInitFile_v4.xlsx" in src:
                if "../config/" in src:
                    cells_with_new_path += 1
                else:
                    cells_with_old_path += 1

        assert cells_with_old_path == 0, (
            f"Tier_Assignment_Salinity: {cells_with_old_path} cell(s) still have "
            f"old CtrlFile path (without ../config/)"
        )
        assert cells_with_new_path >= 2, (
            f"Tier_Assignment_Salinity: expected CtrlFile in at least 2 cells, "
            f"found {cells_with_new_path} with ../config/ prefix"
        )
