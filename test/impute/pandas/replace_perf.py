import pandas as pd
import re
import timeit


def str_replace_perf():
    miscdict = {" isn't ": ' is not '," aren't ":' are not '," wasn't ":' was not '," snevada ":' Sierra Nevada '}
    data=pd.DataFrame({"q1":["beer is ok","beer isn't ok","beer wasn't available"," snevada is good"]})
    def org(printout=False):
        def parse_text(data):
            for key, replacement in miscdict.items():
                data['q1'] = data['q1'].str.replace( key, replacement )
            return data
        data2 = parse_text(data)
        if printout:
            print(data2)
    org(printout=True)
    print("str replace\n", timeit.timeit(org, number=10000))


def reg_replace_perf():
    miscdict = {" isn't ": ' is not '," aren't ":' are not '," wasn't ":' was not '," snevada ":' Sierra Nevada '}
    data=pd.DataFrame({"q1":["beer is ok","beer isn't ok","beer wasn't available"," snevada is good"]})
    def alt1(printout=False):
        data['q1'].replace(miscdict, regex=True, inplace=True)
        if printout:
            print(data)
    alt1(printout=True)
    print("reg replace\n", timeit.timeit(alt1, number=10000))


def reg_compile_perf():
    miscdict = {" isn't ": ' is not '," aren't ":' are not '," wasn't ":' was not '," snevada ":' Sierra Nevada '}
    miscdict_comp = {re.compile(k, re.IGNORECASE): v for k, v in miscdict.items()}
    data=pd.DataFrame({"q1":["beer is ok","beer isn't ok","beer wasn't available"," snevada is good"]})
    def alt3(printout=False):
        def parse_text(text):
            for pattern, replacement in miscdict_comp.items():
                text = pattern.sub(replacement, str(text))
            return text
        data["q1"] = data["q1"].apply(parse_text)
        if printout:
            print(data)
    alt3(printout=True)
    print("reg compile replace\n", timeit.timeit(alt3, number=10000))


str_replace_perf()
reg_replace_perf()
reg_compile_perf()
