from neural_compressor.strategy.st_utils.tuning_space import TuningItem, TuningSpace
from neural_compressor.conf.dotdict import DotDict
from copy import deepcopy
import unittest

op_cap = {
    # op have both weight and activation and support static/dynamic/fp32
    ('op_name1', 'op_type1'): [
        {
            'activation':
                {
                    'dtype': ['uint4'],
                    'quant_mode': 'static',
                    'scheme': ['sym'],
                    'granularity': ['per_channel', 'per_tensor'],
                    'algorithm': ['minmax', 'kl']
                },
            'weight':
                {
                    'dtype': ['int4'],
                    'scheme': ['sym'],
                    'granularity': ['per_channel', 'per_tensor']
                }
        },
        {
            'activation':
                {
                    'dtype': ['int8'],
                    'quant_mode': 'dynamic',
                    'scheme': ['sym'],
                    'granularity': ['per_channel', 'per_tensor'],
                    'algorithm': ['minmax', 'kl']
                },
            'weight':
                {
                    'dtype': ['int8'],
                    'scheme': ['sym'],
                    'granularity': ['per_channel', 'per_tensor']
                }
        },
        {
            'activation':
                {
                    'dtype': 'fp32'
                },
            'weight':
                {
                    'dtype': 'fp32'
                }
        },
    ],
    # # op have both weight and activation and support static/dynamic/fp32
    # ('op_name2', 'op_type1'): [
    #     {
    #         'activation':
    #             {
    #                 'dtype': ['int8'],
    #                 'quant_mode': 'static',
    #                 'scheme': ['sym'],
    #                 'granularity': ['per_channel', 'per_tensor'],
    #                 'algorithm': ['minmax', 'kl']
    #             },
    #         'weight':
    #             {
    #                 'dtype': ['int8'],
    #                 'scheme': ['sym'],
    #                 'granularity': ['per_channel', 'per_tensor']
    #             }
    #     },
    #     {
    #         'activation':
    #             {
    #                 'dtype': ['int8'],
    #                 'quant_mode': 'dynamic',
    #                 'scheme': ['sym'],
    #                 'granularity': ['per_channel', 'per_tensor'],
    #                 'algorithm': ['minmax', 'kl']
    #             },
    #         'weight':
    #             {
    #                 'dtype': ['int8'],
    #                 'scheme': ['sym'],
    #                 'granularity': ['per_channel', 'per_tensor']
    #             }
    #     },
    #     {
    #         'activation':
    #             {
    #                 'dtype': 'fp32'
    #             },
    #         'weight':
    #             {
    #                 'dtype': 'fp32'
    #             }
    #     },
    # ],
    # # op have both weight and activation and support static/fp32
    # ('op_name3', 'op_type2'): [
    #     {
    #         'activation':
    #             {
    #                 'dtype': ['int8'],
    #                 'quant_mode': 'static',
    #                 'scheme': ['sym'],
    #                 'granularity': ['per_channel'],
    #                 'algorithm': ['minmax', 'kl']
    #             },
    #         'weight':
    #             {
    #                 'dtype': ['int8'],
    #                 'scheme': ['sym'],
    #                 'granularity': ['per_channel'],
    #                 'algorithm': ['minmax', 'kl']
    #             }
    #     },
    #     {
    #         'activation':
    #             {
    #                 'dtype': 'fp32'
    #             },
    #         'weight':
    #             {
    #                 'dtype': 'fp32'
    #             }
    #     },
    # ],
    # # op have both weight and activation and support dynamic/fp32
    # ('op_name4', 'op_type3'): [
    #     {
    #         'activation':
    #             {
    #                 'dtype': ['int8'],
    #                 'quant_mode': 'static',
    #                 'scheme': ['sym'],
    #                 'granularity': ['per_channel', 'per_tensor'],
    #                 'algorithm': ['minmax', 'kl']
    #             },
    #     },
    #     {
    #         'activation':
    #             {
    #                 'dtype': ['int8'],
    #                 'quant_mode': 'dynamic',
    #                 'scheme': ['sym'],
    #                 'granularity': ['per_channel', 'per_tensor'],
    #                 'algorithm': ['minmax']
    #             },
    #     },
    #     {
    #         'activation':
    #             {
    #                 'dtype': 'fp32'
    #             },
    #         'weight':
    #             {
    #                 'dtype': 'fp32'
    #             }
    #     },
    # ]

}


class TestTuningSampler(unittest.TestCase):
    def setUp(self) -> None:
        self.capability = {
            'calib': {'calib_sampling_size': [1, 10, 50]},
            'op': deepcopy(op_cap)
        }

        self.optype_wise_user_config = {
            'op_type1': {
                'activation': {
                    'algorithm': ['minmax'],
                    'granularity': ['per_channel', 'per_tensor'],
                }
            }
        }
        self.model_wise_user_config = {
            'activation': {
                'granularity': ['per_channel'],
            }
        }

        self.op_wise_user_config = {
            'op_name4': {
                'activation': {
                    'dtype': ['fp32'],
                }
            }
        }

    def test_tuning_space_creation(self):
        conf = None
        # Test the creation of tuning space 
        tuning_space = TuningSpace(self.capability, conf)
        print(tuning_space.root_item.get_details())
        import pdb
        pdb.set_trace()
        
        # # ops supported static 
        # static_items = tuning_space.query_items_by_quant_mode('static')
        # static_items_name = [item.name for item in static_items]
        # self.assertEqual(static_items_name, list(op_cap.keys()))
        # # ops supported dynamic 
        # dynamic_items = tuning_space.query_items_by_quant_mode('dynamic')
        # dynamic_items_name = [item.name for item in dynamic_items]
        # all_items_name = list(op_cap.keys())
        # all_items_name.remove(('op_name3', 'op_type2'))
        # self.assertEqual(dynamic_items_name, all_items_name)
        # # ops supported fp32 
        # fp32_items = tuning_space.query_items_by_quant_mode('fp32')
        # fp32_items_name = [item.name for item in fp32_items]
        # self.assertEqual(fp32_items_name, list(op_cap.keys()))
        # # all optype
        # self.assertEqual(list(tuning_space.op_type_wise_items.keys()), ['op_type1', 'op_type2', 'op_type3'])

if __name__ == "__main__":
    unittest.main()
