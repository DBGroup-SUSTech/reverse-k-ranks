from recbole.trainer import HyperTuning
from recbole.quick_start import objective_function

hp = HyperTuning(objective_function=objective_function, algo='exhaustive', early_stop=10,
                 max_evals=100, params_file='enmf.test', fixed_config_file_list=['example.yaml'])

# run
hp.run()
# export result to the file
hp.export_result(output_file='hyper_example.result')
# print best parameters
print('best params: ', hp.best_params)
# print best result
print('best result: ')
print(hp.params2result[hp.params2str(hp.best_params)])
