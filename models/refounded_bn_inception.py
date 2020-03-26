from __future__ import print_function, division, absolute_import
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

import pretrainedmodels


# __all__ = ['BNInception', 'bninception']
#
# pretrained_settings = {
#     'bninception': {
#         'imagenet': {
#             'url': 'https://www.dropbox.com/s/3cvod6kzwluijcw/BNInception-9baff57459f5a1744.pth?dl=1',
#             'input_space': 'BGR',
#             'input_size': 224,
#             'input_range': [0, 255],
#             'mean': [104, 117, 128],
#             'std': [1, 1, 1],
#             'num_classes': 1000
#         },
#         'kinetics': {
#             'url': 'https://www.dropbox.com/s/gx4u7itoyygix0c/BNInceptionKinetics-47f0695e.pth?dl=1',
#             'input_space': 'BGR',
#             'input_size': 224,
#             'input_range': [0, 255],
#             'mean': [104, 117, 128],  # [96.29023126, 103.16065604, 110.63666788]
#             'std': [1, 1, 1],  # [40.02898126, 37.88248729, 38.7568578],
#             'num_classes': 400
#         }
#     },
# }


class Inception_Block(nn.Module):
    def __init__(self, component_list):
        super(Inception_Block, self).__init__()
        self.component_list = component_list

    def forward(self, x):
        component_num = len(self.component_list)
        for component_idx in range(component_num):
            if component_idx == 0:
                output = self.component_list[component_idx](x)
            else:
                output = torch.cat((output, self.component_list[component_idx](x)), 1)

        return output


class refounded_BNInception(nn.Module):
    def __init__(self, num_classes=1000):
        super(refounded_BNInception, self).__init__()
        model_name = 'bninception'  # could be fbresnet152 or inceptionresnetv2
        pretrained_model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        modules = list(pretrained_model.children())

        self._build_features(modules, num_classes)

    def forward(self, x):
        # # stage1
        # pool1_3x3_s2_out = self.block_1(x)
        # # stage2
        # pool2_3x3_s2_out = self.block_2(pool1_3x3_s2_out)

        pool2_3x3_s2_out = self.shallow_block(x)

        # stage3
        inception_3a_output_out = self.inception_3a_blk(pool2_3x3_s2_out)
        inception_3b_output_out = self.inception_3b_blk(inception_3a_output_out)
        inception_3c_output_out = self.inception_3c_blk(inception_3b_output_out)

        inception_4a_output_out = self.inception_4a_blk(inception_3c_output_out)
        inception_4b_output_out = self.inception_4b_blk(inception_4a_output_out)
        inception_4c_output_out = self.inception_4c_blk(inception_4b_output_out)
        inception_4d_output_out = self.inception_4d_blk(inception_4c_output_out)
        inception_4e_output_out = self.inception_4e_blk(inception_4d_output_out)

        inception_5a_output_out = self.inception_5a_blk(inception_4e_output_out)
        inception_5b_output_out = self.inception_5b_blk(inception_5a_output_out)

        x = self.logits(inception_5b_output_out)
        return x

    def logits(self, features):
        x = self.global_pool(features)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


    def _build_features(self, modules, num_classes):
        self.shallow_block = nn.Sequential(*modules[:11])

        self.inception_3a_1x1_blk = nn.Sequential(*modules[11: 14])
        self.inception_3a_3x3_reduce_blk = nn.Sequential(*modules[14: 20])
        self.inception_3a_double_3x3_reduce_blk = nn.Sequential(*modules[20: 29])
        self.inception_3a_pool_blk = nn.Sequential(*modules[29: 33])

        self.inception_3a_blk = Inception_Block([
            self.inception_3a_1x1_blk,
            self.inception_3a_3x3_reduce_blk,
            self.inception_3a_double_3x3_reduce_blk,
            self.inception_3a_pool_blk
        ])


        self.inception_3b_1x1_blk = nn.Sequential(*modules[33: 36])
        self.inception_3b_3x3_reduce_blk = nn.Sequential(*modules[36: 42])
        self.inception_3b_double_3x3_reduce_blk = nn.Sequential(*modules[42: 51])
        self.inception_3b_pool_blk = nn.Sequential(*modules[51: 55])

        self.inception_3b_blk = Inception_Block([
            self.inception_3b_1x1_blk,
            self.inception_3b_3x3_reduce_blk,
            self.inception_3b_double_3x3_reduce_blk,
            self.inception_3b_pool_blk
        ])


        self.inception_3c_3x3_reduce_blk = nn.Sequential(*modules[55: 61])
        self.inception_3c_double_3x3_reduce_blk = nn.Sequential(*modules[61: 70])
        self.inception_3c_pool = modules[70]

        self.inception_3c_blk = Inception_Block([
            self.inception_3c_3x3_reduce_blk,
            self.inception_3c_double_3x3_reduce_blk,
            self.inception_3c_pool
        ])


        self.inception_4a_1x1_blk = nn.Sequential(*modules[71: 74])
        self.inception_4a_3x3_reduce_blk = nn.Sequential(*modules[74: 80])
        self.inception_4a_double_3x3_reduce_blk = nn.Sequential(*modules[80: 89])
        self.inception_4a_pool_blk = nn.Sequential(*modules[89: 93])

        self.inception_4a_blk = Inception_Block([
            self.inception_4a_1x1_blk,
            self.inception_4a_3x3_reduce_blk,
            self.inception_4a_double_3x3_reduce_blk,
            self.inception_4a_pool_blk
        ])


        self.inception_4b_1x1_blk = nn.Sequential(*modules[93: 96])
        self.inception_4b_3x3_reduce_blk = nn.Sequential(*modules[96: 102])
        self.inception_4b_double_3x3_reduce_blk = nn.Sequential(*modules[102: 111])
        self.inception_4b_pool_blk = nn.Sequential(*modules[111: 115])

        self.inception_4b_blk = Inception_Block([
            self.inception_4b_1x1_blk,
            self.inception_4b_3x3_reduce_blk,
            self.inception_4b_double_3x3_reduce_blk,
            self.inception_4b_pool_blk
        ])


        self.inception_4c_1x1_blk = nn.Sequential(*modules[115: 118])
        self.inception_4c_3x3_reduce_blk = nn.Sequential(*modules[118: 124])
        self.inception_4c_double_3x3_reduce_blk = nn.Sequential(*modules[124: 133])
        self.inception_4c_pool_blk = nn.Sequential(*modules[133: 137])

        self.inception_4c_blk = Inception_Block([
            self.inception_4c_1x1_blk,
            self.inception_4c_3x3_reduce_blk,
            self.inception_4c_double_3x3_reduce_blk,
            self.inception_4c_pool_blk
        ])


        self.inception_4d_1x1_blk = nn.Sequential(*modules[137: 140])
        self.inception_4d_3x3_reduce_blk = nn.Sequential(*modules[140: 146])
        self.inception_4d_double_3x3_reduce_blk = nn.Sequential(*modules[146: 155])
        self.inception_4d_pool = nn.Sequential(*modules[155: 159])

        self.inception_4d_blk = Inception_Block([
            self.inception_4d_1x1_blk,
            self.inception_4d_3x3_reduce_blk,
            self.inception_4d_double_3x3_reduce_blk,
            self.inception_4d_pool
        ])


        self.inception_4e_3x3_reduce_blk = nn.Sequential(*modules[159: 165])
        self.inception_4e_double_3x3_reduce_blk = nn.Sequential(*modules[165: 174])
        self.inception_4e_pool = modules[174]

        self.inception_4e_blk = Inception_Block([
            self.inception_4e_3x3_reduce_blk,
            self.inception_4e_double_3x3_reduce_blk,
            self.inception_4e_pool
        ])


        self.inception_5a_1x1_blk = nn.Sequential(*modules[175: 178])
        self.inception_5a_3x3_reduce_blk = nn.Sequential(*modules[178: 184])
        self.inception_5a_double_3x3_reduce_blk = nn.Sequential(*modules[184: 193])
        self.inception_5a_pool_blk = nn.Sequential(*modules[193: 197])

        self.inception_5a_blk = Inception_Block([
            self.inception_5a_1x1_blk,
            self.inception_5a_3x3_reduce_blk,
            self.inception_5a_double_3x3_reduce_blk,
            self.inception_5a_pool_blk
        ])


        self.inception_5b_1x1_blk = nn.Sequential(*modules[197: 200])
        self.inception_5b_3x3_reduce_blk = nn.Sequential(*modules[200: 206])
        self.inception_5b_double_3x3_reduce_blk = nn.Sequential(*modules[206: 215])
        self.inception_5b_pool_blk = nn.Sequential(*modules[215: 219])

        self.inception_5b_blk = Inception_Block([
            self.inception_5b_1x1_blk,
            self.inception_5b_3x3_reduce_blk,
            self.inception_5b_double_3x3_reduce_blk,
            self.inception_5b_pool_blk
        ])


        self.global_pool = nn.AvgPool2d(7, stride=1, padding=0, ceil_mode=True, count_include_pad=True)
        self.fc = nn.Linear(1024, num_classes)


if __name__ == '__main__':
    model = refounded_BNInception()
