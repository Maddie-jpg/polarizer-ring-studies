import numpy as np
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import pandas as pd
import pyarrow as pa

#
#  Save the tracking data in a parquet file, one for the data and one for the header info.
#  The header file name is generated automatically with an extra extension.
#
#  mon: the tracking object from Xsuite
#  header: is the header information for the tracking in the form of a dictionary with parameters
#  filename: the base name of the file, the extension will be added
#
def saveTrackingData(mon, header, filename):

    no_turns_eff = len(mon.state.sum(axis=0))
    df = pd.DataFrame({'pol_y': np.mean(mon.spin_y, axis=0),
                   'pol_x': np.mean(mon.spin_x, axis=0),
                   'pol_z': np.mean(mon.spin_z, axis=0),
                   'std_pol_y': np.std(mon.spin_y, axis=0),
                   'std_pol_x': np.std(mon.spin_x, axis=0),
                   'std_pol_z': np.std(mon.spin_z, axis=0),
                   'no_part' : mon.state.sum(axis=0),
                   'p0_gev' : np.mean(mon.p0c, axis=0)*1E-9,
                   'gamma0' : np.mean(mon.gamma0, axis=0),
                   'x':  np.mean(mon.x, axis=0),
                   'y': np.mean(mon.y, axis=0),
                   'delta': np.mean(mon.delta, axis=0),
                   'std_x': np.std(mon.x, axis=0),
                   'std_y': np.std(mon.y, axis=0),
                   'std_delta': np.std(mon.delta, axis=0)
                   },
                  index=range(no_turns_eff))

    table_df = pa.Table.from_pandas(df)
    f_name = filename + '.pq'
    pq.write_table(table_df, f_name)

    df2 = pd.DataFrame([header])
    table_df2 = pa.Table.from_pandas(df2)
    header_f_name = filename + '.header.pq'
    pq.write_table(table_df2, header_f_name)

    print('Saved tracking data in files ', f_name, ' and ', header_f_name)
    
    return
#
#  Save the tracking data in a parquet file, one for the data and one for the header info.
#  The header file name is generated automatically with an extra extension.
#
#  mon: the tracking object from Xsuite
#  filename: the base name of the file, the extension will be added
#
def saveTrackingDataNoHeader(mon, filename):

    no_turns_eff = len(mon.state.sum(axis=0))
    df = pd.DataFrame({'pol_y': np.mean(mon.spin_y, axis=0),
                   'pol_x': np.mean(mon.spin_x, axis=0),
                   'pol_z': np.mean(mon.spin_z, axis=0),
                   'std_pol_y': np.std(mon.spin_y, axis=0),
                   'std_pol_x': np.std(mon.spin_x, axis=0),
                   'std_pol_z': np.std(mon.spin_z, axis=0),
                   'no_part' : mon.state.sum(axis=0),
                   'p0_gev' : np.mean(mon.p0c, axis=0)*1E-9,
                   'gamma0' : np.mean(mon.gamma0, axis=0),
                   'x':  np.mean(mon.x, axis=0),
                   'y': np.mean(mon.y, axis=0),
                   'delta': np.mean(mon.delta, axis=0),
                   'std_x': np.std(mon.x, axis=0),
                   'std_y': np.std(mon.y, axis=0),
                   'std_delta': np.std(mon.delta, axis=0)
                   },
                  index=range(no_turns_eff))

    table_df = pa.Table.from_pandas(df)
    f_name = filename + '.pq'
    pq.write_table(table_df, f_name)

    print('Saved tracking data in file ', f_name)
    
    return

#
#  Save the tracking data in a parquet file, one for the data and one for the header info.
#  The header file name is generated automatically with an extra extension.
#
#  data: the tracking dict with the data
#  header: is the header information for the tracking in the form of a dictionary with parameters
#  filename: the base name of the file, the extension will be added
#
def saveTrackingDatainDict(data, header = None, filename = 'datafile'):
 
    df = pd.DataFrame([data])
    table_df = pa.Table.from_pandas(df)
    f_name = filename + '.pq'
    pq.write_table(table_df, f_name)
    print('Saved data in ', f_name)

    if header != None:
        df2 = pd.DataFrame([header])
        table_df2 = pa.Table.from_pandas(df2)
        header_f_name = filename + '.header.pq'
        pq.write_table(table_df2, header_f_name)
        print('Saved header in ', header_f_name)
  
    return

#
#  Loads the tracking data for the data file and header file name (optional)
#  Returns 2 dataframes with the tracking data and the header data
#

def loadTrackingData(filename, headername = None):

    table = pq.read_table(filename)
    df = table.to_pandas()
 
    if headername != None:
        tableh = pq.read_table(headername)
        dfheader = tableh.to_pandas()
    else:
        hdict = {'header' : "empty"}
        dfheader = pd.DataFrame([hdict])

    return df,dfheader 
