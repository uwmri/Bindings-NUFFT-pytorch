# Author: Alban Gossard
# Last modification: 2022/23/08

import torch
import torchkbnufft as tkbn
from nufftbindings.basenufft import *


class Nufft(baseNUFFT):
    def _set_dims(self):
        if self.ndim==2:
            self.XX, self.XY = torch.meshgrid(self.xx, self.xy)
            self.zx = self.xx.view(-1,1).repeat(1, self.nx).view(-1,1)
            self.zy = self.xy.view(1,-1).repeat(self.ny, 1).view(-1,1)
            self.nufft_ob = tkbn.KbNufft((self.nx,self.ny)).to(self.device)
            self.nufft_adj_ob = tkbn.KbNufftAdjoint((self.nx,self.ny)).to(self.device)
        elif self.ndim==3:
            self.XX, self.XY, self.XZ = torch.meshgrid(self.xx, self.xy, self.xz)
            self.zx = self.xx.view(-1,1,1).repeat(1, self.nx, self.nx).view(-1,1)
            self.zy = self.xy.view(1,-1,1).repeat(self.ny, 1, self.ny).view(-1,1)
            self.zz = self.xy.view(1,1,-1).repeat(self.nz, self.nz, 1).view(-1,1)
            self.nufft_ob = tkbn.KbNufft((self.nx,self.ny,self.nz)).to(self.device)
            self.nufft_adj_ob = tkbn.KbNufftAdjoint((self.nx,self.ny,self.nz)).to(self.device)

    def precompute(self, xi):
        self.xiprecomputed = xi.clone()
        self.precomputedTrig = True

    def _forward2D(self, f, xi, smaps=None):
        self.test_xi(xi)
        ndim = len(f.shape)
        iscpx = f.is_complex()
        # if ndim != 4 and not iscpx or ndim != 3 and iscpx:
        #     raise Exception("Error: f should have 4 dimensions: batch, nx, ny, r/i or 3 dimensions: batch, nx, ny (complex dtype)")
        if iscpx:
            if ndim == 3:
                f = f[:,None].type(self.torch_cpxdtype) # batch,1,nx,ny
            else:
                f = f.type(self.torch_cpxdtype) # batch,1,nx,ny

            xi = xi.permute(1, 0).type(self.torch_dtype)

            if smaps is None:
                y = self.nufft_ob(f, xi, smaps)[:, 0]
            else:
                y = self.nufft_ob(f, xi, smaps=smaps)

        else:
            f = f[:, None].type(self.torch_dtype) # batch,nx,ny,r/i
            xi = xi.permute(1, 0).type(self.torch_dtype)
            y = self.nufft_ob(f, xi, smaps)[:,0]

        return y

    def _adjoint2D(self, y, xi, smaps=None):
        self.test_xi(xi)
        ndim = len(y.shape)
        iscpx = y.is_complex()
        # if ndim != 3 and not iscpx or ndim != 2 and iscpx:
        #     raise Exception("Error: y should have 3 dimensions: batch, K, r/i or 2 dimensions: batch, K (complex dtype)")
        if iscpx and smaps is None:
            y = y[:,None].type(self.torch_cpxdtype) # batch, c, K
        elif iscpx and smaps is not None:
            y = y.type(self.torch_cpxdtype) # batch, c, K
        else:
            # wrong but don't care for now
            y = y[:,None].type(self.torch_dtype) # batch,K,r/i
        xi = xi.permute(1,0).type(self.torch_dtype)
        if smaps is not None:
            f = self.nufft_adj_ob(y, xi, smaps=smaps)
        else:
            f = self.nufft_adj_ob(y, xi)[:,0]
        return f


    def _backward_forward2D(self, f, g, xi, smaps=None):
        # print(f'_backward_forward2D: input f shape{f.shape}')
        # print(f'_backward_forward2D: input g shape{g.shape}')

        self.test_xi(xi)
        ndim = len(f.shape)
        iscpx = f.is_complex()
        grad = torch.zeros(xi.shape, dtype=self.torch_dtype, device=self.device)
        # if ndim != 4 and not iscpx or ndim != 3 and iscpx:
        #     raise Exception("Error: f should have 4 dimensions: batch, nx, ny, r/i or 3 dimensions: batch, nx, ny (complex dtype)")
        if iscpx and smaps is None:
            f = f[:,None].type(self.torch_cpxdtype)         # batch,1,nx,ny
            xi = xi.permute(1,0).type(self.torch_dtype)
            g = g[:,None].type(self.torch_cpxdtype)         # batch, 1, klength
            vec_fx = torch.mul(self.XX[None,None], f)       #(batch, 1, nx, ny)
            vec_fy = torch.mul(self.XY[None,None], f)
            # print(f'_backward_forward2D: vec_fx shape{vec_fx.shape}')

            tmp = self.nufft_ob(vec_fx, xi)[:,0]
            grad[:,0] = ( torch.mul(tmp.imag, g[:,0].real) - torch.mul(tmp.real, g[:,0].imag) ).sum(axis=0)
            tmp = self.nufft_ob(vec_fy, xi)[:,0]
            grad[:,1] = ( torch.mul(tmp.imag, g[:,0].real) - torch.mul(tmp.real, g[:,0].imag) ).sum(axis=0)

        elif iscpx and smaps is not None:

            f = f.type(self.torch_cpxdtype)    # batch,c,nx,ny
            xi = xi.permute(1, 0).type(self.torch_dtype)
            g = g.type(self.torch_cpxdtype)    # batch, c, klength
            vec_fx = torch.mul(self.XX[None, None], f)  # (batch, 1, nx, ny)
            vec_fy = torch.mul(self.XY[None, None], f)

            tmp = self.nufft_ob(vec_fx, xi, smaps=smaps)
            grad[:,0] = ( torch.mul(tmp.imag, g.real) - torch.mul(tmp.real, g.imag) ).sum(axis=(0,1))
            tmp = self.nufft_ob(vec_fy, xi, smaps=smaps)
            grad[:,1] = ( torch.mul(tmp.imag, g.real) - torch.mul(tmp.real, g.imag) ).sum(axis=(0,1))

        else:
            f = f[:,None].type(self.torch_dtype) # batch,nx,ny,r/i
            xi = xi.permute(1,0).type(self.torch_dtype)
            g = g[:,None].type(self.torch_dtype) # batch,K,r/i
            #                          batch,coil,nx,ny,r/i
            vec_fx = torch.mul(self.XX[None,None,...,None], f)
            vec_fy = torch.mul(self.XY[None,None,...,None], f)

            tmp = self.nufft_ob(vec_fx, xi)[:,0]
            grad[:,0] = ( torch.mul(tmp[...,1], g[:,0,...,0]) - torch.mul(tmp[...,0], g[:,0,...,1]) ).sum(axis=0)
            tmp = self.nufft_ob(vec_fy, xi)[:,0]
            grad[:,1] = ( torch.mul(tmp[...,1], g[:,0,...,0]) - torch.mul(tmp[...,0], g[:,0,...,1]) ).sum(axis=0)

        return grad

    def _backward_adjoint2D(self, y, g, xi, smaps=None):
        self.test_xi(xi)
        ndim = len(y.shape)
        iscpx = y.is_complex()
        grad = torch.zeros(xi.shape, dtype=self.torch_dtype, device=self.device)
        # if ndim != 3 and not iscpx or ndim != 2 and iscpx:
        #     raise Exception("Error: y should have 3 dimensions: batch, K, r/i or 2 dimensions: batch, K (complex dtype)")
        if iscpx and smaps is None:
            y = y[:,None].type(self.torch_cpxdtype)                 # BATCH, 1, klength
            xi = xi.permute(1,0).type(self.torch_dtype)
            g = g[:,None].type(self.torch_cpxdtype)

            vecx_grad_output = torch.mul(self.XX[None,None], g)     # BATCH, 1, NX, NY
            vecy_grad_output = torch.mul(self.XY[None,None], g)

            # print(f'_backward_adjoint2D single coil: vecx_grad_output shape{vecx_grad_output.shape}')
            # print(f'_backward_adjoint2D single coil: y shape{y.shape}')

            tmp = self.nufft_ob(vecx_grad_output, xi)[:,0]          # batch, klength
            # print(f'_backward_adjoint2D single coil: tmp shape{tmp.shape}')

            grad[:,0] = ( torch.mul(tmp.imag, y[:,0].real) - torch.mul(tmp.real, y[:,0].imag) ).sum(axis=0)
            tmp = self.nufft_ob(vecy_grad_output, xi)[:,0]
            grad[:,1] = ( torch.mul(tmp.imag, y[:,0].real) - torch.mul(tmp.real, y[:,0].imag) ).sum(axis=0)

        elif iscpx and smaps is not None:

            y = y.type(self.torch_cpxdtype)     #(batch, coil, klength)
            xi = xi.permute(1,0).type(self.torch_dtype)
            g = g.type(self.torch_cpxdtype)     #(batch, 1, nx, ny)

            vecx_grad_output = torch.mul(self.XX[None,None], g)     #(batch, 1, nx, ny)
            vecy_grad_output = torch.mul(self.XY[None,None], g)

            tmp = self.nufft_ob(vecx_grad_output, xi, smaps=smaps)      #(batch, coil, klength)

            grad[:,0] = ( torch.mul(tmp.imag, y.real) - torch.mul(tmp.real, y.imag) ).sum(axis=(0,1))
            tmp = self.nufft_ob(vecy_grad_output, xi, smaps=smaps)
            grad[:,1] = ( torch.mul(tmp.imag, y.real) - torch.mul(tmp.real, y.imag) ).sum(axis=(0,1))

        else:
            y = y[:,None].type(self.torch_dtype) # batch,K,r/i
            xi = xi.permute(1,0).type(self.torch_dtype)
            g = g[:,None].type(self.torch_dtype) # batch,nx,ny,r/i
            #                          batch,coil,nx,ny,r/i
            vecx_grad_output = torch.mul(self.XX[None,None,...,None], g)
            vecy_grad_output = torch.mul(self.XY[None,None,...,None], g)

            tmp = self.nufft_ob(vecx_grad_output, xi)[:,0]
            grad[:,0] = ( torch.mul(tmp[...,1], y[:,0,...,0]) - torch.mul(tmp[...,0], y[:,0,...,1]) ).sum(axis=0)
            tmp = self.nufft_ob(vecy_grad_output, xi)[:,0]
            grad[:,1] = ( torch.mul(tmp[...,1], y[:,0,...,0]) - torch.mul(tmp[...,0], y[:,0,...,1]) ).sum(axis=0)

        return grad

    def _forward3D(self, f, xi, smaps=None):
        self.test_xi(xi)
        ndim = len(f.shape)
        iscpx = f.is_complex()
        # if ndim != 5 and not iscpx or ndim != 4 and iscpx:
        #     raise Exception("Error: f should have 5 dimensions: batch, nx, ny, nz, r/i or 4 dimensions (complex")
        if iscpx:
            if ndim == 4:
                f = f[:,None].type(self.torch_cpxdtype) # batch, 1,nx,ny, nz
            else:
                f = f.type(self.torch_cpxdtype) # batch,1,nx,ny

        else:
            f = f[:,None].type(self.torch_dtype) # batch,nx,ny,nz,r/i
        xi = xi.permute(1,0).type(self.torch_dtype)

        if smaps is None:
            y = self.nufft_ob(f, xi)[:,0]
        else:
            y = self.nufft_ob(f, xi, smaps=smaps)

        return y


    def _adjoint3D(self, y, xi, smaps=None):
        self.test_xi(xi)
        ndim = len(y.shape)
        iscpx = y.is_complex()
        # if ndim != 3 and not iscpx or ndim != 2 and iscpx:
        #     raise Exception("Error: y should have 3 dimensions: batch, K, r/i or 2 dimensions: batch, K(complex)")
        if iscpx and smaps is None:
            y = y[:,None].type(self.torch_cpxdtype) # batch,K

        elif iscpx and smaps is not None:
            y = y.type(self.torch_cpxdtype) # batch, c, K

        else:
            y = y[:,None].type(self.torch_dtype) # batch,K,r/i
        xi = xi.permute(1,0).type(self.torch_dtype)

        if smaps is not None:
            f = self.nufft_adj_ob(y, xi, smaps=smaps)
        else:
            f = self.nufft_adj_ob(y, xi)[:, 0]
        return f

    def _backward_forward3D(self, f, g, xi, smaps=None):
        self.test_xi(xi)
        ndim = len(f.shape)
        # print(f'backward_forward3D: f shape {f.shape}')

        iscpx = f.is_complex()
        grad = torch.zeros(xi.shape, dtype=self.torch_dtype, device=self.device)
        if ndim != 5 and not iscpx or ndim != 4 and iscpx:
            raise Exception("Error: f should have 3 dimensions: batch, nx, ny, nz, r/i or 4 dimensions: batch, nx, ny, nz (complex dtype)")
        if iscpx and smaps is None:
            f = f[:,None].type(self.torch_cpxdtype) # batch,nx,ny
            xi = xi.permute(1,0).type(self.torch_dtype)
            g = g[:,None].type(self.torch_cpxdtype) # batch,K
            #                          batch,coil,nx,ny
            vec_fx = torch.mul(self.XX[None,None], f)
            # print(f'backward_forward3D: vec_fx shape {vec_fx.shape}')
            # print(f'backward_forward3D: g shape {g.shape}')
            # print(f'backward_forward3D: xi shape {xi.shape}')
            # print(f'backward_forward3D: f shape {f.shape}')

            vec_fy = torch.mul(self.XY[None,None], f)
            vec_fz = torch.mul(self.XZ[None,None], f)

            tmp = self.nufft_ob(vec_fx, xi)[:, 0]
            # print(f'backward_forward3D: tmp shape {tmp.shape}')
            grad[:, 0] = (torch.mul(tmp.imag, g[:, 0].real) - torch.mul(tmp.real, g[:, 0].imag)).sum(axis=0)
            tmp = self.nufft_ob(vec_fy, xi)[:, 0]
            grad[:, 1] = (torch.mul(tmp.imag, g[:, 0].real) - torch.mul(tmp.real, g[:, 0].imag)).sum(axis=0)
            tmp = self.nufft_ob(vec_fz, xi)[:, 0]
            grad[:, 2] = (torch.mul(tmp.imag, g[:, 0].real) - torch.mul(tmp.real, g[:, 0].imag)).sum(axis=0)

        elif iscpx and smaps is not None:

            f = f.type(self.torch_cpxdtype)    # batch,c,nx,ny
            xi = xi.permute(1, 0).type(self.torch_dtype)
            g = g.type(self.torch_cpxdtype)    # batch, c, klength
            vec_fx = torch.mul(self.XX[None, None], f)  # (batch, 1, nx, ny)
            vec_fy = torch.mul(self.XY[None, None], f)
            vec_fz = torch.mul(self.XZ[None,None], f)

            tmp = self.nufft_ob(vec_fx, xi, smaps=smaps)
            grad[:,0] = ( torch.mul(tmp.imag, g.real) - torch.mul(tmp.real, g.imag) ).sum(axis=(0,1))
            tmp = self.nufft_ob(vec_fy, xi, smaps=smaps)
            grad[:,1] = ( torch.mul(tmp.imag, g.real) - torch.mul(tmp.real, g.imag) ).sum(axis=(0,1))
            tmp = self.nufft_ob(vec_fz, xi, smaps=smaps)
            grad[:,2] = ( torch.mul(tmp.imag, g.real) - torch.mul(tmp.real, g.imag) ).sum(axis=(0,1))

        else:
            f = f[:,None].type(self.torch_dtype) # batch,nx,ny,nz,r/i
            xi = xi.permute(1,0).type(self.torch_dtype)
            g = g[:,None].type(self.torch_dtype) # batch,K,r/i
            #                          batch,coil,nx,ny,nz,r/i
            vec_fx = torch.mul(self.XX[None,None,...,None], f)
            vec_fy = torch.mul(self.XY[None,None,...,None], f)
            vec_fz = torch.mul(self.XZ[None,None,...,None], f)

            tmp = self.nufft_ob(vec_fx, xi)[:,0]
            grad[:,0] = ( torch.mul(tmp[...,1], g[:,0,...,0]) - torch.mul(tmp[...,0], g[:,0,...,1]) ).sum(axis=0)
            tmp = self.nufft_ob(vec_fy, xi)[:,0]
            grad[:,1] = ( torch.mul(tmp[...,1], g[:,0,...,0]) - torch.mul(tmp[...,0], g[:,0,...,1]) ).sum(axis=0)
            tmp = self.nufft_ob(vec_fz, xi)[:,0]
            grad[:,2] = ( torch.mul(tmp[...,1], g[:,0,...,0]) - torch.mul(tmp[...,0], g[:,0,...,1]) ).sum(axis=0)

        return grad

    def _backward_adjoint3D(self, y, g, xi, smaps=None):
        self.test_xi(xi)
        ndim = len(y.shape)
        iscpx = y.is_complex()
        grad = torch.zeros(xi.shape, dtype=self.torch_dtype, device=self.device)
        # if ndim != 3 and not iscpx or ndim != 2 and iscpx:
        #     raise Exception("Error: y should have 3 dimensions: batch, K, r/i")
        if iscpx and smaps is None:
            y = y[:,None].type(self.torch_cpxdtype) # batch,K
            xi = xi.permute(1,0).type(self.torch_dtype)
            g = g[:,None].type(self.torch_cpxdtype) # batch,nx,ny
            #                          batch,coil,nx,ny
            vecx_grad_output = torch.mul(self.XX[None,None], g)
            vecy_grad_output = torch.mul(self.XY[None,None], g)
            vecz_grad_output = torch.mul(self.XZ[None,None], g)

            tmp = self.nufft_ob(vecx_grad_output, xi)[:,0]
            grad[:,0] = ( torch.mul(tmp.imag, y[:,0].real) - torch.mul(tmp.real, y[:,0].imag) ).sum(axis=0)
            tmp = self.nufft_ob(vecy_grad_output, xi)[:,0]
            grad[:,1] = ( torch.mul(tmp.imag, y[:,0].real) - torch.mul(tmp.real, y[:,0].imag) ).sum(axis=0)
            tmp = self.nufft_ob(vecz_grad_output, xi)[:,0]
            grad[:,2] = ( torch.mul(tmp.imag, y[:,0].real) - torch.mul(tmp.real, y[:,0].imag) ).sum(axis=0)
            # print(f'backward_forward3D: vec_fx shape {vecx_grad_output.shape}')
            # print(f'backward_forward3D: g shape {g.shape}')
            # print(f'backward_forward3D: xi shape {xi.shape}')
            # print(f'backward_forward3D: f shape {y.shape}')
            # print(f'backward_forward3D: tmp shape {tmp.shape}')
        elif iscpx and smaps is not None:

            y = y.type(self.torch_cpxdtype)     #(batch, coil, klength)
            xi = xi.permute(1,0).type(self.torch_dtype)
            g = g.type(self.torch_cpxdtype)     #(batch, 1, nx, ny)

            vecx_grad_output = torch.mul(self.XX[None,None], g)     #(batch, 1, nx, ny)
            vecy_grad_output = torch.mul(self.XY[None,None], g)
            vecz_grad_output = torch.mul(self.XY[None,None], g)

            tmp = self.nufft_ob(vecx_grad_output, xi, smaps=smaps)      #(batch, coil, klength)
            grad[:,0] = ( torch.mul(tmp.imag, y.real) - torch.mul(tmp.real, y.imag) ).sum(axis=(0,1))
            tmp = self.nufft_ob(vecy_grad_output, xi, smaps=smaps)
            grad[:,1] = ( torch.mul(tmp.imag, y.real) - torch.mul(tmp.real, y.imag) ).sum(axis=(0,1))
            tmp = self.nufft_ob(vecz_grad_output, xi, smaps=smaps)
            grad[:,2] = ( torch.mul(tmp.imag, y.real) - torch.mul(tmp.real, y.imag) ).sum(axis=(0,1))

        else:
            y = y[:,None].type(self.torch_dtype) # batch,K,r/i
            xi = xi.permute(1,0).type(self.torch_dtype)
            g = g[:,None].type(self.torch_dtype) # batch,nx,ny,nz,r/i
            #                          batch,coil,nx,ny,nz,r/i
            vecx_grad_output = torch.mul(self.XX[None,None,...,None], g)
            vecy_grad_output = torch.mul(self.XY[None,None,...,None], g)
            vecz_grad_output = torch.mul(self.XY[None,None,...,None], g)

            tmp = self.nufft_ob(vecx_grad_output, xi)[:,0]
            grad[:,0] = ( torch.mul(tmp[...,1], y[:,0,...,0]) - torch.mul(tmp[...,0], y[:,0,...,1]) ).sum(axis=0)
            tmp = self.nufft_ob(vecy_grad_output, xi)[:,0]
            grad[:,1] = ( torch.mul(tmp[...,1], y[:,0,...,0]) - torch.mul(tmp[...,0], y[:,0,...,1]) ).sum(axis=0)
            tmp = self.nufft_ob(vecz_grad_output, xi)[:,0]
            grad[:,1] = ( torch.mul(tmp[...,1], y[:,0,...,0]) - torch.mul(tmp[...,0], y[:,0,...,1]) ).sum(axis=0)

        return grad


nufft=Nufft()



class FClass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xi, f, smaps=None):
        ctx.save_for_backward(xi, f, smaps)
        output = nufft.forward(f, xi, smaps)
        return output
    @staticmethod
    def backward(ctx, grad_output):
        xi, f, smaps = ctx.saved_tensors
        grad_input = nufft.backward_forward(f, grad_output, xi, smaps)
        grad_input_f = nufft.adjoint(grad_output, xi, smaps)
        return grad_input, grad_input_f, None

class FtClass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xi, y, smaps=None):
        ctx.save_for_backward(xi, y, smaps)
        output = nufft.adjoint(y, xi, smaps)
        return output
    @staticmethod
    def backward(ctx, grad_output):
        xi, y, smaps = ctx.saved_tensors
        grad_input = nufft.backward_adjoint(y, grad_output, xi, smaps)      # (klength, 2)

        # print(f'gradoutput shape {grad_output.shape}')
        # print(f'gradinput shape {grad_input.shape}')

        grad_input_y = nufft.forward(grad_output, xi, smaps)    # gradoutputshape  (batch, 1, nx, ny), need to remove that extra coil dimension in nufft.forward

        return grad_input, grad_input_y, None

forward = FClass.apply
adjoint = FtClass.apply
