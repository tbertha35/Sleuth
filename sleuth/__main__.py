import sys

from .cli import main

# The guard matters: worker processes (tokenizer training, sharding) start
# with "spawn", which re-imports this module. Without it, every worker would
# re-run the command.
if __name__ == "__main__":
    sys.exit(main())
