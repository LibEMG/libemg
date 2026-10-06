"""``python -m libemg.session``.

The command line lives behind its own module so running it does not re-execute a
copy of the annotations module the package has already imported -- the screen
guided training panel imports it, so by the time a terminal asks for the command
it is always loaded.
"""

import sys

from libemg.session import main

if __name__ == "__main__":
    sys.exit(main())
