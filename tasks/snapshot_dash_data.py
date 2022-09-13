#! /usr/bin/env python

import dash_lib as dl
import datetime
import pandas as pd
import time


print('hello')

# import pdb; pdb.set_trace()
dd = dl.DashData()

#dd.run()
df = dd.get_latest('dash_logo_renewals')
print(df.to_string(index=False))




