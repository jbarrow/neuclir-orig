import _jsonnet
import sys

if len(sys.argv) != 3:
    print('Usage:\npython conf.py [INPUT].jsonnet [OUTPUT].json')
    sys.exit(0)

f_inp = sys.argv[1]
f_out = sys.argv[2]

js = _jsonnet.evaluate_file(f_inp)
fp = open(f_out, 'w')
fp.write(js)
fp.close()
