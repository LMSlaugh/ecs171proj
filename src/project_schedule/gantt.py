'''Create and update the gantt chart for current project.

Update progress.txt and run this script.
Plot accessible at https://plot.ly/~richardli068/0.
'''

import os
import sys

import pandas as pd

import plotly
import plotly.plotly as py
import plotly.figure_factory as ff


plotly.tools.set_credentials_file(username='richardli068',
                                  api_key='qk3at7xIMWCr18p7jOn5')


def main():
    df = pd.read_csv(os.path.dirname(sys.argv[0]) + '/progress.txt',
                     delim_whitespace=True,
                     names=['Task', 'Start', 'Finish', 'Complete'],
                     header=0)
    fig = ff.create_gantt(df,
                          colors=['#FF0000', '#008000'],
                          index_col='Complete',
                          show_colorbar=True)
    py.iplot(fig, filename='progress', overwrite=True)


if __name__ == "__main__":
    main()
