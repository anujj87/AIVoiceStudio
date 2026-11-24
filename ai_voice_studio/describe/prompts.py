"""The prompts an audio description run uses.

The descriptions a vision-language model writes are only usable by a blind or
low-vision listener when they are *short, objective and true*.  The accessibility
literature is explicit that the prompt, not the model size, is what turns
"describe everything" into "describe what matters" - and that verbosity and
invented detail are the two complaints listeners raise most.  So the guideline
prompt below is a first-class part of the feature, not a user preference.

It distils the recurring professional audio-description guidelines (the kind the
VideoA11y study collected from 154 industry standards) into instructions a small
open model can follow: say what matters, say it objectively, say it once, never
invent, never talk over the dialogue.
"""

from __future__ import annotations

#: The system prompt every description request carries.
GUIDELINE_PROMPT = (
    "You are an audio describer for blind and low-vision viewers.  You watch a "
    "short piece of a video and describe only what a sighted viewer would need "
    "in order to follow the story and the action.\n"
    "\n"
    "Rules:\n"
    "1. Describe in the present tense, in one or two short sentences.\n"
    "2. Say what matters most first: who is doing what, where, and how it "
    "looks or changes.  Skip the unimportant.\n"
    "3. Be objective.  Report what is visible and audible, not how you feel "
    "about it and not what you assume it means.\n"
    "4. Never invent.  If you cannot see something clearly, do not describe it. "
    "Do not guess names, brands, places or identities from appearance.\n"
    "5. Name a person only when the video itself tells you who they are.\n"
    "6. Do not describe on-screen text word for word unless it is essential; "
    "say that text appears and what it is about.\n"
    "7. Do not mention that you are an AI, do not address the viewer, and do "
    "not add commentary, warnings or summaries.\n"
    "8. If nothing significant changes in this piece, answer exactly: "
    "\"No significant visual change.\"\n"
    "\n"
    "Write only the description itself, in plain prose."
)

#: The instruction one description **window** carries (see
#: :data:`ai_voice_studio.constants.AD_WINDOW_SECONDS`).
#:
#: It asks for the piece and its change, because a window is short enough that
#: "what is happening now" and "what is different" are the same question - and
#: because a model that is only asked to describe, with no memory of the last
#: description, answers with the scene rather than with what has just happened in
#: it.  The worker adds the previous description to this; the instruction itself
#: has to make sense on the opening window, which has none.  It replaced a plain
#: "describe the small piece of video" instruction, which is what a run asked for
#: when one call covered a whole chunk.
WINDOW_INSTRUCTION = (
    "Describe this short piece of the video in one or two short sentences, as "
    "the next thing a blind or low-vision viewer hears. Name what is on screen "
    "and what has just happened or changed in it."
)

#: Used when the description has to fit a known silent gap.
def gap_instruction(max_seconds: float) -> str:
    """A chunk instruction bounded by the silence it has to fit into."""
    words = max(4, round(max(2.0, float(max_seconds)) * 2.6))
    return (
        f"Describe this piece of video in at most {words} words, so the "
        "description can be spoken inside a pause without covering any "
        "dialogue."
    )


#: Shown in the UI as the one-line reason the prompt matters.
PROMPT_NOTE = (
    "Audio description quality comes mostly from the guideline prompt: it is "
    "what keeps descriptions short, objective and free of invented detail."
)
