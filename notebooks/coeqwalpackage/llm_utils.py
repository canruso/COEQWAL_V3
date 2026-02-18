"""LLM utilities for COEQWAL scenario analysis.

Provides functions to send plot images to an LLM (Claude) for
automated interpretation and to generate executive summaries
from collected observations.
"""

import base64
import os


def _get_client(api_key=None):
    """Return an Anthropic client, creating one if needed.

    Resolution order for API key:
      1. Explicit ``api_key`` argument
      2. ANTHROPIC_API_KEY environment variable
      3. Interactive prompt via getpass
    """
    from anthropic import Anthropic

    if api_key is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key is None:
        from getpass import getpass
        api_key = getpass("Enter Anthropic API key: ")

    return Anthropic(api_key=api_key)


_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def analyze_plot(image_path, prompt, model="claude-sonnet-4-20250514",
                 max_tokens=500, api_key=None):
    """Send a plot image to Claude and return the observation text.

    Parameters
    ----------
    image_path : str
        Path to a PNG/JPG plot file.
    prompt : str
        What to analyze — e.g. "Describe the key differences between
        scenarios in this exceedance plot."
    model : str
        Anthropic model ID.
    max_tokens : int
        Maximum response length.
    api_key : str, optional
        Override for API key (otherwise env / prompt).

    Returns
    -------
    str
        The model's observation text.
    """
    client = _get_client(api_key)

    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    ext = os.path.splitext(image_path)[1].lower()
    media_type = _MEDIA_TYPES.get(ext, "image/png")

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data,
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }],
    )

    return response.content[0].text


def generate_summary(observations_text, scenario_info,
                     model="claude-sonnet-4-20250514", max_tokens=1500,
                     api_key=None):
    """Generate an executive summary from collected plot observations.

    Parameters
    ----------
    observations_text : str
        Formatted string of all individual plot observations.
    scenario_info : str
        Description of scenarios being compared.
    model : str
        Anthropic model ID.
    max_tokens : int
        Maximum response length.
    api_key : str, optional
        Override for API key.

    Returns
    -------
    str
        Executive summary text.
    """
    client = _get_client(api_key)

    prompt = (
        "You are analyzing CalSim3 water system model outputs comparing "
        "multiple California water management scenarios.\n\n"
        f"Scenarios:\n{scenario_info}\n\n"
        "Below are observations from individual plot analyses. Write a "
        "concise executive summary (3-5 paragraphs) covering:\n"
        "1. Overall patterns — which scenarios consistently perform "
        "better or worse?\n"
        "2. Key tradeoffs — where does improving one outcome worsen "
        "another?\n"
        "3. Notable findings — any unexpected results or surprises?\n"
        "4. Recommendations — which scenarios merit further "
        "investigation and why?\n\n"
        f"Observations:\n{observations_text}"
    )

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.content[0].text
