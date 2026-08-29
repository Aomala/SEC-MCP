"""omala — easter egg.

    python omala.py          sign it
    OMALA=1 <anything>       sign it on import

Nothing here is load-bearing. Delete it and the world keeps turning.

    — Austin Omala · github.com/Aomala
"""

import os
import sys

SIGNATURE = (
    "                  \x1b[97m▄▄███▀▀██▄\x1b[0m\n"
    "              \x1b[97m▄▄██▀▀       ▀█▄\x1b[0m\n"
    "           \x1b[97m▄██▀▀            ▀█\x1b[0m\n"
    "        \x1b[97m▄▄██▀    ▄▄██        █\x1b[38;5;196m█\x1b[0m\n"
    "      \x1b[97m▄██▀     ▄███▀        ██\x1b[38;5;196m█\x1b[0m\n"
    "    \x1b[97m▄██▀    ▄▄▀▄█▀          █\x1b[38;5;196m█\x1b[0m\n"
    "   \x1b[97m██▀    ▄█▀ ▄█▀          █\x1b[38;5;196m█▀\x1b[0m\n"
    " \x1b[97m▄█▀    ▄▀▀  ██           █\x1b[38;5;196m██                            \x1b[97m▄███\x1b[0m\n"
    "\x1b[97m▄█▀   ▄\x1b[38;5;196m▀   \x1b[97m▄██          ▄█\x1b[38;5;196m█▀                            \x1b[97m████▀\x1b[0m\n"
    "\x1b[97m█▀ ▄\x1b[38;5;196m▄\x1b[97m▀    ▄██          ▄█\x1b[38;5;196m█▀                           \x1b[97m▄██▄█\x1b[0m\n"
    "\x1b[97m▀█\x1b[38;5;196m█▀      \x1b[97m██         ▄█\x1b[38;5;196m█▀                             \x1b[97m███▀\x1b[0m\n"
    " \x1b[38;5;196m█▀      \x1b[97m██        ▄█\x1b[38;5;196m██\x1b[97m▄▀▀█  ▄██ ▄▄          ▄▄      ███     ▄▄▄\x1b[0m\n"
    "\x1b[38;5;196m█▀      \x1b[97m██       ▄\x1b[38;5;196m███\x1b[97m█▀  ███▀███▀██▄▄▄    ▄██▀██    ██▀   ▄██▀██           ▄\x1b[0m\n"
    "\x1b[38;5;196m▀█      \x1b[97m██    \x1b[38;5;196m▄███\x1b[97m█▀   ▄██▀ ██▀ ▄██▀▀██▄▄███▄███ ▄▄███  ▄███▄███        ▄▄▀\x1b[0m\n"
    "  \x1b[38;5;196m▀▀   \x1b[97m███ ▀\x1b[38;5;196m██\x1b[97m██▀     ▄█▀  ██▀  ▀▀   ▀▀▀▀██▀  ██▀▀  ██▄█▀▀█▀▀ ▀█▄▄▄▄▄▄▀▀▀\x1b[0m\n"
    "        \x1b[97m████▀▀         \x1b[38;5;196m▄▄▄▄▄▄▄▄▄████▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀  ▄▄▄▄▄▄\x1b[0m\n"
    "            \x1b[38;5;196m▄▄▄▄▄███▀▀▀▀▀▀▀                                         \x1b[97m▀\x1b[38;5;196m▀▀\x1b[0m\n"
    "     \x1b[38;5;196m▄▄▄█▀▀▀▀▀\x1b[0m\n"
    "  \x1b[38;5;196m▄▀▀\x1b[0m\n"
)

PLAIN = """\
                  ▄▄███▀▀██▄
              ▄▄██▀▀       ▀█▄
           ▄██▀▀            ▀█
        ▄▄██▀    ▄▄██        ██
      ▄██▀     ▄███▀        ███
    ▄██▀    ▄▄▀▄█▀          ██
   ██▀    ▄█▀ ▄█▀          ██▀
 ▄█▀    ▄▀▀  ██           ███                            ▄███
▄█▀   ▄▀   ▄██          ▄██▀                            ████▀
█▀ ▄▄▀    ▄██          ▄██▀                           ▄██▄█
▀██▀      ██         ▄██▀                             ███▀
 █▀      ██        ▄███▄▀▀█  ▄██ ▄▄          ▄▄      ███     ▄▄▄
█▀      ██       ▄████▀  ███▀███▀██▄▄▄    ▄██▀██    ██▀   ▄██▀██           ▄
▀█      ██    ▄████▀   ▄██▀ ██▀ ▄██▀▀██▄▄███▄███ ▄▄███  ▄███▄███        ▄▄▀
  ▀▀   ███ ▀████▀     ▄█▀  ██▀  ▀▀   ▀▀▀▀██▀  ██▀▀  ██▄█▀▀█▀▀ ▀█▄▄▄▄▄▄▀▀▀
        ████▀▀         ▄▄▄▄▄▄▄▄▄████▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀  ▄▄▄▄▄▄
            ▄▄▄▄▄███▀▀▀▀▀▀▀                                         ▀▀▀
     ▄▄▄█▀▀▀▀▀
  ▄▀▀"""


def sign(stream=None):
    """Write the signature. Falls back to plain blocks when there is no tty."""
    out = stream or sys.stdout
    color = getattr(out, "isatty", lambda: False)() and os.environ.get("TERM") != "dumb"
    out.write((SIGNATURE if color else PLAIN) + "\n\n")
    out.write("        austin omala  \u00b7  github.com/Aomala\n")
    out.flush()


if os.environ.get("OMALA"):
    sign()

if __name__ == "__main__":
    sign()
