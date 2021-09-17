import os
from utils import cmd_input 
from reporting import (generate_tables, 
                        generate_residual_maps, 
                        generate_segmenatation_maps, 
                        mvtec_eval,
                        hera_eval)

def report():
    '''
        Checks if the results files have been correctly generated 
            and produces desired plots
    '''

    elif ((cmd_input.args.data == 'HERA') or
            (cmd_input.args.data == 'LOFAR')):
        hera_eval(cmd_input.args)

    else:
        raise Exception('Can only generate results on\
                            HERA/LOFAR')

if __name__ == '__main__':
    report()
