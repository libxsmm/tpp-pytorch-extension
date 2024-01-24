
HERE=$(cd "$(dirname "$0")" && pwd -P)
gcc -fPIC -shared -ldl -o ./omptrace.so ${HERE}/omptrace.c
