from dask_quantumtinkerer import Cluster
from dask_quantumtinkerer import cluster_options
import dask.bag as db
import dask.array as da
import dask
import xarray as xr
import traitlets
import numpy as np
import itertools as it
import paramiko
from pathlib import Path
import datetime
import os


class dask_io(traitlets.HasTraits):
    params = traitlets.Dict()
    OUTPUT_DIR = traitlets.Unicode()
    N_output = traitlets.Int(default_value=1)

    def __init__(self, f, params, OUTPUT_DIR, N_output=None):
        """Instantiates the input arrays, the paths and directories on local and remote.

        Arguments:
            f {[func]} -- function to compute. Must return a single value or a tuple
            params {[dict]} -- dictionary of the values to pass to f
            OUTPUT_DIR {[str]} -- output directory path (relative to the current working directory)

        Keyword Arguments:
            N_output {[int]} -- number of outputs f returns. If not given, f will determined by computing it
        """
        self.params = params
        self.OUTPUT_DIR = OUTPUT_DIR
        self.f = f

        values = list(self.params.values())
        self.args = np.array(list(it.product(*values)))
        self.shapes = [len(values[i]) for i in range(len(values))]

        if N_output is None:
            try:
                N_output_test = np.shape(f(*self.args[0]))
                if N_output_test == ():
                    pass
                else:
                    self.N_output = N_output_test[0]
            except Exception as e:
                print('Either the function or the parameter input is ill-defined')
                raise

        if self.N_output > 1:
            self.shapes = [*self.shapes, self.N_output]

        print('Setting up directory on remote and local...')

        # Obtain current directory path
        BASE_DIR_IO = "/home/tinkerer/"
        BASE_DIR_HPC05 = "/home/kostasvilkelis/work"
        CWD = os.getcwd().split(BASE_DIR_IO)[1]
        self.LOCAL_PATH = Path(BASE_DIR_IO)/Path(CWD)
        self.REMOTE_PATH = Path(BASE_DIR_HPC05)/Path(CWD)

        # Create data directory
        DATE = Path(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S/"))
        DIRECTORY = OUTPUT_DIR/DATE
        self.TEMP_DIR = DIRECTORY/Path('temp_data')
        L_TEMP_DIR = self.LOCAL_PATH/self.TEMP_DIR
        L_TEMP_DIR.mkdir(parents=True, exist_ok=True)

        # Set up file paths
        PARAM_AR = self.TEMP_DIR/Path('params.npy')
        self.TEMP_DATA = self.TEMP_DIR/Path('temp_data*.hdf')
        self.XR_DATA = self.LOCAL_PATH/DIRECTORY/Path('XR_DATA')

        np.save(self.LOCAL_PATH/PARAM_AR, self.args)

        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy)
        self.ssh.connect('hpc05', username='kostasvilkelis', password=None)
        self.ssh.exec_command(f'mkdir -p {self.REMOTE_PATH/self.TEMP_DIR}')
        sftp = self.ssh.open_sftp()
        sftp.put(str(self.LOCAL_PATH/PARAM_AR), str(self.REMOTE_PATH/PARAM_AR))
        sftp.close()
        print('...Done!')

    worker_memory = traitlets.Float()
    nodes = traitlets.Int()
    partition_size = traitlets.Int()

    def cluster_calc(self, worker_memory=2, nodes=5, partition_size=1):
        """Starts the computation on cluster via dask. Saves results in HDF format on the cluster and transfers it back to io.

        Keyword Arguments:
            worker_memory {float} -- memory per each worker (GB) (default: {2})
            nodes {int} -- Number of nodes (default: {5})
        """
        self.worker_memory = worker_memory
        self.nodes = nodes
        self.partition_size = partition_size

        f = self.f

        def wrapped_fn(args):
            return f(*args)

        options = cluster_options()
        options.worker_memory = self.worker_memory
        options.extra_path = str(self.REMOTE_PATH)
        print('Connecting to the Cluster...')
        with Cluster(options) as cluster:
            print('...Connected! The dashboard is given at the address:')
            print('http://io.quantumtinkerer.tudelft.nl/user/Kostas/proxy/' +
                  cluster.dashboard_link[17:])
            print('Connecting to workers and beginning calculation')
            cluster.scale(self.nodes)
            client = cluster.get_client()
            # Computation
            baged_input = db.from_sequence(self.args, partition_size=self.partition_size)
            data = baged_input.map(wrapped_fn).to_dataframe().to_hdf(
                self.REMOTE_PATH/self.TEMP_DATA, 'group')
            # job_list = [da.from_array(wrapped_fn(i)) for i in args]
            # data = da.stack(job_list, axis=0)
            # This should be better
            
            
        print('Calculation finished. Transfering data to local...')
        self.ssh.connect('hpc05', username='kostasvilkelis', password=None)
        sftp = self.ssh.open_sftp()
        rfiles = sftp.listdir(str(self.REMOTE_PATH/self.TEMP_DIR))
        for i in rfiles:
            sftp.get(str(self.REMOTE_PATH/self.TEMP_DIR/Path(i)),
                     str(self.LOCAL_PATH/self.TEMP_DIR/Path(i)))
        sftp.close()
        print('...transfer complete!')

    def to_xr(self):
        """Computes the xarray object from the HDF results and saves it.

        Returns:
            [xarray obj] -- results
        """
        loaded_data = dask.dataframe.read_hdf(
            str(self.LOCAL_PATH/self.TEMP_DATA), 'group').to_dask_array(lengths=True)
        shaped_data = da.reshape(loaded_data, self.shapes)
        if self.N_output == 1:
            xr_array = xr.DataArray(data=shaped_data,
                                    dims=self.params.keys(),
                                    coords=self.params)
        else:
            xr_array = xr.DataArray(data=shaped_data,
                                    dims=[*self.params.keys(), 'type'],
                                    coords={**self.params, 'type': np.arange(self.N_output)})
        xr_array.to_netcdf(self.XR_DATA)
        return xr_array
