import sys

from auditwheel.main import main
from auditwheel.policy import _POLICIES as POLICIES

# libs are loaded dynamically; do not include them
for p in POLICIES:
    p['lib_whitelist'].append('libmemx.so')
    p['lib_whitelist'].append('libmx_accl.so')
    p['lib_whitelist'].append('libmx_accl.so.2')

if __name__ == "__main__":
    sys.exit(main())
