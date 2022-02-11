s = 'z�br�'

import locale
ENCODING = locale.getpreferredencoding()
s.encode(ENCODING).decode('utf-8')

print(s)