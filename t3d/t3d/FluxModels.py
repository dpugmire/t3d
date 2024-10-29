from collections import OrderedDict
from t3d.Logbook import info


class FluxModelDict():
    '''
    This library contains model functions for fluxes.

    + there is an analytic ReLU model called Flux_model
    + there is an anlaytic model from Barnes' thesis
    + there is the GX flux model
    '''

    def __init__(self, inputs, grid, time):
        from t3d.flux_models.GX import GX_FluxModel
        from t3d.flux_models.ReLU import ReLU_FluxModel
        from t3d.flux_models.Diffusive import Diffusive_FluxModel
        from t3d.flux_models.ChangHintonNeoclassical import ChangHintonNeo_FluxModel
        from t3d.flux_models.KNOSOS import KNOSOS_FluxModel
        from t3d.flux_models.SFINCS import SFINCS_FluxModel
        from t3d.flux_models.PowerBalance import PowerBalance_FluxModel

        model_list_params = inputs.get('model', {})
        self.N_models = len(model_list_params)
        self.models_dict = OrderedDict()

        models = {
            "GX": GX_FluxModel,
            "gx": GX_FluxModel,
            "ReLU": ReLU_FluxModel,
            "relu": ReLU_FluxModel,
            "Diffusive": Diffusive_FluxModel,
            "diffusive": Diffusive_FluxModel,
            "ChangHinton": ChangHintonNeo_FluxModel,
            "KNOSOS": KNOSOS_FluxModel,
            "knosos": KNOSOS_FluxModel,
            "SFINCS": SFINCS_FluxModel,
            "sfincs": SFINCS_FluxModel,
            "PowerBalance": PowerBalance_FluxModel
        }

        if type(model_list_params) is list:
            for pars in model_list_params:
                my_model = pars.get('model', 'GX')
                model = models[my_model](pars, grid, time)
                label = model.label
                # ensure unique key for dictionary in case of duplicate labels
                while label in self.models_dict.keys():
                    label = label + "_"
                self.models_dict[label] = model
        else:
            my_model = model_list_params.get('model', 'GX')
            self.models_dict[my_model] = models[my_model](model_list_params, grid, time)

        # check for compatibility of sqrt(2) v_th conventions
        self.vt_sqrt_2 = None
        for model in self.get_models_list():
            if model.vt_sqrt_2 is not None and self.vt_sqrt_2 is None:
                self.vt_sqrt_2 = model.vt_sqrt_2
            elif model.vt_sqrt_2 is not None and self.vt_sqrt_2 is not None:
                assert model.vt_sqrt_2 == self.vt_sqrt_2, "Error: cannot have flux models with different v_th sqrt(2) conventions"

        # Print logfile for flux models
        info('\n  Flux model log files')
        for model in self.get_models_list():
            info(f'    {model.label}: {model.logfile_info()}')

        if self.vt_sqrt_2 is None:
            self.vt_sqrt_2 = False

    def __getitem__(self, s):
        '''
        accessor that allows a particular element (FluxModel) of the models_dict dictionary to be accessed via models[s] == models.models_dict[s]
        '''
        return self.models_dict[s]

    def get_models_list(self):
        '''
        return list of all elements of models_dict. each element is a FluxModel (sub)class member
        '''
        return self.models_dict.values()


# base class
class FluxModel():

    # base class constructor, should be called in derived class
    # constructor by super().__init__()
    def __init__(self, inputs, grid, time):
        self.N_fluxtubes = len(grid.midpoints)
        self.rho = grid.midpoints
        self.grid = grid
        self.time = time

        self.vt_sqrt_2 = None

    def compute_fluxes(self, species, geo, norms):
        assert False, "Error: no compute_fluxes function for this flux model."

    def logfile_info(self) -> str:
        return ''

    def final_info(self):
        pass

    def collect_results(self, species, geo, norms):
        pass
