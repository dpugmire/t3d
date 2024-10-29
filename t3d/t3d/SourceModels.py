
from collections import OrderedDict
from t3d.Logbook import info


class SourceModelDict():
    '''
    This library contains model functions for fluxes.
    '''

    def __init__(self, inputs, grid, time):
        from t3d.source_models.BEAMS3D import BEAMS3D_SourceModel

        source_list_params = inputs.get('source', {})
        self.N_sources = len(source_list_params)
        self.sources_dict = OrderedDict()

        sources = {
            "BEAMS3D": BEAMS3D_SourceModel,
        }

        if type(source_list_params) is list:
            for pars in source_list_params:
                my_source = pars.get('source')
                self.sources_dict[my_source] = sources[my_source](pars, grid, time)
        elif len(source_list_params) > 0:
            my_source = source_list_params.get('source')
            self.sources_dict[my_source] = sources[my_source](source_list_params, grid, time)

        # Print logfile for source models
        info('\n  Source model log files')
        for model in self.get_sources_list():
            info(f'    {model.label}: {model.logfile_info()}')

    def __getitem__(self, s):
        '''
        accessor that allows a particular element (SourceModel) of the sources_dict dictionary to be accessed via sources[s] == sources.sources_dict[s]
        '''
        return self.sources_dict[s]

    def get_sources_list(self):
        '''
        return list of all elements of sources_dict. each element is a SourceModel (sub)class member
        '''

        return self.sources_dict.values()


# base class
class SourceModel():

    # base class constructor, should be called in derived class
    # constructor by super().__init__()
    def __init__(self, inputs, grid, time):
        self.N_radial = len(grid.rho)
        self.rho = grid.rho
        self.grid = grid
        self.time = time

    def compute_sources(self, species, geo, norms):
        assert False, "Error: no compute_sources function for this source model."

    def final_info(self):
        pass

    def collect_results(self, species, geo, norms):
        pass
