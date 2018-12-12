from fastai.torch_imports import *
from fastai.conv_learner import *
from torch.nn.utils.spectral_norm import spectral_norm

class ConvBlock(nn.Module):
    def __init__(self, ni:int, no:int, ks:int=3, stride:int=1, pad:int=None, actn:bool=True, 
            bn:bool=True, bias:bool=True, sn:bool=False, leakyReLu:bool=False, self_attention:bool=False,
            inplace_relu:bool=True):
        super().__init__()   
        if pad is None: pad = ks//2//stride

        if sn:
            layers = [spectral_norm(nn.Conv2d(ni, no, ks, stride, padding=pad, bias=bias))]
        else:
            layers = [nn.Conv2d(ni, no, ks, stride, padding=pad, bias=bias)]
        if actn:
            layers.append(nn.LeakyReLU(0.2, inplace=inplace_relu)) if leakyReLu else layers.append(nn.ReLU(inplace=inplace_relu)) 
        if bn:
            layers.append(nn.BatchNorm2d(no))
        if self_attention:
            layers.append(SelfAttention(no, 1))

        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)


class UpSampleBlock(nn.Module):
    @staticmethod
    def _conv(ni:int, nf:int, ks:int=3, bn:bool=True, sn:bool=False, leakyReLu:bool=False):
        layers = [ConvBlock(ni, nf, ks=ks, sn=sn, bn=bn, actn=False, leakyReLu=leakyReLu)]
        return nn.Sequential(*layers)

    @staticmethod
    def _icnr(x:torch.Tensor, scale:int=2):
        init=nn.init.kaiming_normal_
        new_shape = [int(x.shape[0] / (scale ** 2))] + list(x.shape[1:])
        subkernel = torch.zeros(new_shape)
        subkernel = init(subkernel)
        subkernel = subkernel.transpose(0, 1)
        subkernel = subkernel.contiguous().view(subkernel.shape[0],
                                                subkernel.shape[1], -1)
        kernel = subkernel.repeat(1, 1, scale ** 2)
        transposed_shape = [x.shape[1]] + [x.shape[0]] + list(x.shape[2:])
        kernel = kernel.contiguous().view(transposed_shape)
        kernel = kernel.transpose(0, 1)
        return kernel

    def __init__(self, ni:int, nf:int, scale:int=2, ks:int=3, bn:bool=True, sn:bool=False, leakyReLu:bool=False):
        super().__init__()
        layers = []
        assert (math.log(scale,2)).is_integer()

        for i in range(int(math.log(scale,2))):
            layers += [UpSampleBlock._conv(ni, nf*4,ks=ks, bn=bn, sn=sn, leakyReLu=leakyReLu), MYPixelShuffle(2)]
            if bn:
                layers += [nn.BatchNorm2d(nf)]

            ni = nf
                       
        self.sequence = nn.Sequential(*layers)
        self._icnr_init()
        
    def _icnr_init(self):
        conv_shuffle = self.sequence[0][0].seq[0]
        kernel = UpSampleBlock._icnr(conv_shuffle.weight)
        conv_shuffle.weight.data.copy_(kernel)
    
    def forward(self, x):
        #items = list(self.sequence.children())
        #x = items[0](x)
        #x = items[1](x)
        #print(items[1])
        #return x
        return self.sequence(x)#


class UnetBlock(nn.Module):
    def __init__(self, up_in:int , x_in:int , n_out:int, bn:bool=True, sn:bool=False, leakyReLu:bool=False, 
            self_attention:bool=False, inplace_relu:bool=True):
        super().__init__()
        up_out = x_out = n_out//2
        self.x_conv  = ConvBlock(x_in,  x_out,  ks=1, bn=False, actn=False, sn=sn, inplace_relu=inplace_relu)
        self.tr_conv = UpSampleBlock(up_in, up_out, 2, bn=bn, sn=sn, leakyReLu=leakyReLu)
        self.relu = nn.LeakyReLU(0.2, inplace=inplace_relu) if leakyReLu else nn.ReLU(inplace=inplace_relu)
        out_layers = []
        if bn: 
            out_layers.append(nn.BatchNorm2d(n_out))
        if self_attention:
            out_layers.append(SelfAttention(n_out))
        self.out = nn.Sequential(*out_layers)
        
        
    def forward(self, up_p:int, x_p:int):
        up_p = self.tr_conv(up_p)
        x_p = self.x_conv(x_p)#
        x = torch.cat([up_p,x_p], dim=1)#
        x = self.relu(x)#
        return self.out(x)#
        #return up_p

class SaveFeatures():
    features=None
    def __init__(self, m:nn.Module): 
        self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): 
        self.features = output
    def remove(self): 
        self.hook.remove()

class SelfAttention(nn.Module):
    def __init__(self, in_channel:int, gain:int=1):
        super().__init__()
        print("in channel")
        print(in_channel)
        self.query = self._spectral_init(nn.Conv1d(in_channel, in_channel // 8, 1),gain=gain)
        self.key = self._spectral_init(nn.Conv1d(in_channel, in_channel // 8, 1),gain=gain)
        self.value = self._spectral_init(nn.Conv1d(in_channel, in_channel, 1), gain=gain)
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def _spectral_init(self, module:nn.Module, gain:int=1):
        nn.init.kaiming_uniform_(module.weight, gain)
        if module.bias is not None:
            module.bias.data.zero_()

        return spectral_norm(module)
    
    def multi(self, batch1: torch.Tensor, batch2: torch.Tensor):
        batch3 = torch.zeros(batch1.shape[0], batch2.shape[1])
        for i in range(batch1.shape[0]):
            for j in range(batch2.shape[1]):
                for k in range(batch1.shape[1]):
                    value2 = batch1[i][k] * batch2[k][j]
                    batch3[i][j] += value2.float()
        return batch3
    
    def forward(self, input:torch.Tensor):
        print("forward SELFATTENTION")
        shape = input.shape
        in_channel = 1024
        query2D = nn.Conv2d(in_channel, in_channel // 8, (1, 1))
        query2DShape = self.query.weight.shape
        query2DShape2 = self.query.bias.shape
        query2D.weight = nn.Parameter(self.query.weight.view(query2DShape[0], query2DShape[1], query2DShape[2], 1))
        query2D.bias = nn.Parameter(self.query.bias.view(query2DShape2[0]))
        
        key2D = nn.Conv2d(in_channel, in_channel // 8, (1, 1))
        key2DShape = self.key.weight.shape
        key2DShape2 = self.key.bias.shape
        key2D.weight = nn.Parameter(self.key.weight.view(key2DShape[0], key2DShape[1], key2DShape[2], 1))
        key2D.bias = nn.Parameter(self.key.bias.view(key2DShape2[0]))
        
        value2D = nn.Conv2d(in_channel, in_channel, (1, 1))
        value2DShape = self.value.weight.shape
        value2DShape2 = self.value.bias.shape
        value2D.weight = nn.Parameter(self.value.weight.view(value2DShape[0], value2DShape[1], value2DShape[2], 1))
        value2D.bias = nn.Parameter(self.value.bias.view(value2DShape2[0]))
        
        flatten2D = input.view(int(shape[0]), int(shape[1]), 14400, 1)
        print(flatten2D.shape)
        
        query = query2D(flatten2D)
        query = query.permute(0, 2, 1, 3)
        key = key2D(flatten2D)
        value = value2D(flatten2D)
        
        query_key = torch.bmm(query.view(1, 14400, 128), key.view(1, 128, 14400))
        #query_zeros = torch.zeros([14400, 14400]) 
        #query_key = torch.addmm(query_zeros, query.view(14400, 128), key.view(128, 14400))
        #query_key = self.multi(query.view(14400, 128), key.view(128, 14400))
        ####query_key = torch.mm(query.view(14400, 128), key.view(128, 14400))
        #query_key = torch.matmul(query.view(14400, 128), key.view(128, 14400))
        #query_key = query_key.view(1, int(query_key.shape[0]), int(query_key.shape[1]))
        #query_key = query_key.view(1, int(14400), int(14400))

        ###TODEBG
        #query_key = input
        ###TODEBG
        attn = query_key
        
        for x in range(int(attn.shape[1])):
            attn[:, x] = F.softmax(attn[:, x])
        
        attn = torch.bmm(value.view(1, 1024, 14400), attn.view(1, 14400, 14400))
        #attn_zeros = torch.zeros([1024, 14400]) 
        #attn = torch.addmm(attn_zeros, value.view(1024, 14400), attn.view(14400, 14400))
        #attn = self.multi(value.view(1024, 14400), attn.view(14400, 14400))
        ####attn = torch.mm(value.view(1024, 14400), attn.view(14400, 14400))
        #attn = torch.matmul(value.view(1024, 14400), attn.view(14400, 14400))
        
        
        attn = attn.view(int(shape[0]), int(shape[1]), int(shape[2]), int(shape[3]))
        out = float(self.gamma) * attn + input
        return out


class MYPixelShuffle(nn.Module):
    
    def __init__(self, upscale_factor:int):
        super().__init__()
        self.upscale_factor = upscale_factor
    
    def forward(self, input:torch.Tensor):
        batch_size = int(input.shape[0])
        channels = int(input.shape[1])
        in_height = int(input.shape[2])
        in_width = int(input.shape[3])
        up_scl = int(self.upscale_factor)
        channels //= up_scl ** 2
        out_height, out_width = in_height * up_scl, in_width * up_scl
        
        input_view = input.contiguous().view(int(batch_size), int(channels), int(up_scl), int(up_scl), int(in_height), int(in_width))
        shuffle_out = input_view.permute(0, 1, 4, 2, 5, 3)
        return shuffle_out.contiguous().view(int(batch_size), int(channels), int(out_height), int(out_width))
        
