import numpy as np

#ds^2 = -(1+2A)*dt^2 + 2a*Bi*dx^i*dt + a^2*((1+2C)*delij + hij)*dx^i*dx^j

g = ['-(1+2A)','2aB1','2aB2','2aB3; 2aB1','a^2(1+2C + h11)','a^2h12','a^2h13; 2aB2','a^2h21','a^2*(1+2C + h22)','a^2h23; 2aB3','a^2h31','a^2h32','a^2*(1+2C + h33)']

delg00 = ['-2Ax0','-2Ax1','-2Ax2','-2Ax3']

delg0i = ['2ax0B0 + 2aB0x0','2aB0x1','2aB0x2','2aB0x3; 2ax0B1 + 2aB1x0','2aB1x1','2aB1x2','2aB1x3; 2ax0B2 + 2aB2x0','2aB2x1','2aB2x2','2aB2x3; 2ax0B3 + 2aB3x0','2aB3x1','2aB3x2','2aB3x3']

delgijdelx0 = ['2a(1+2C + h00) + a^2(2Cx0+h00x0)','2ah01 + a^2h01x0','2ah02 + a^2h02x0','2ah03 + a^2h03x0; 2ah10 + a^2h10x0','2a(1+2C + h11) + a^2(2Cx0+h11x0)','2ah12 + a^2h12x0','2ah13 + a^2h13x0; 2ah20 + a^2h20x0','2ah21 + a^2h21x0','2a(1+2C + h22) + a^2(2Cx0+h22x0)','2ah23 + a^2h23x0; 2ah30 + a^2h30x0','2ah31 + a^2h31x0','2ah32 + a^2h32x0','2a(1+2C + h33) + a^2(2Cx0+h33x0)']

delgijdelx1 = ['a^2(2Cx1+h00x1)','a^2h01x1','a^2h02x1','a^2h03x1; a^2h10x1','a^2(2Cx1+h11x1)','a^2h12x1','a^2h13x1; a^2h20x1','a^2h21x1','a^2(2Cx1+h22x1)','a^2h23x1; a^2h30x1','a^2h31x1','a^2h32x1','a^2(2Cx1+h33x1)']
 
delgijdelx2 = ['a^2(2Cx2+h00x2)','a^2h01x2','a^2h02x2','a^2h03x2; a^2h10x2','a^2(2Cx2+h11x2)','a^2h12x2','a^2h13x2; a^2h20x2','a^2h21x2','a^2(2Cx2+h22x2)','a^2h23x2; a^2h30x2','a^2h31x2','a^2h32x2','a^2(2Cx2+h33x2)']
 
delgijdelx3 = ['a^2(2Cx3+h00x3)','a^2h01x3','a^2h02x3','a^2h03x3; a^2h10x3','a^2(2Cx3+h11x3)','a^2h12x3','a^2h13x3; a^2h20x3','a^2h21x3','a^2(2Cx3+h22x3)','a^2h23x3; a^2h30x3','a^2h31x3','a^2h32x3','a^2(2Cx3+h33x3)']

Lambda[i,j,k] = 0.5*(1/g[i,m])*(delg[m,k,l] + delg[l,m,k] - delg[k,l,m])

Lambda[i,j,k] = 0.5*(1/g[i,0])*(delg[0,k,l] + delg[l,0,k] - delg[k,l,0]) + 0.5*(1/g[i,1])*(delg[1,k,l] + delg[l,1,k] - delg[k,l,1]) + 0.5*(1/g[i,2])*(delg[2,k,l] + delg[l,2,k] - delg[k,l,2]) + 0.5*(1/g[i,3])*(delg[3,k,l] + delg[l,3,k] - delg[k,l,3])

Lambda0[j,k] = 0.5*(1/g[0,m])*(delg[m,k,l] + delg[l,m,k] - delg[k,l,m])


Lambda0 = np.




































