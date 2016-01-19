#!/usr/bin/env python
# encoding: utf-8

"""
Test problem demonstrating a Sedov blast wave problem.
A spherical step function energy perturbation is initialized at the center of
the domain.  This creates an expanding shock wave.

This problem evolves the 3D Euler equations.
The primary variables are:
    density (rho), x,y, and z momentum (rho*u,rho*v,rho*w), and energy.
"""

import logging

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from scipy import integrate
from clawpack import riemann
from clawpack.riemann.euler_3D_constants import density, x_momentum, \
                y_momentum, z_momentum, energy, num_eqn

timeVec1 = PETSc.Vec().createWithArray([0])
timeVec2 = PETSc.Vec().createWithArray([0])
timeVec3 = PETSc.Vec().createWithArray([0])

gamma = 1.4 # Ratio of Specific Heats

x0 = 0.0; y0 = 0.0; z0 = 0.0 # Sphere location
rmax = 0.10 # Radius of Sedov Sphere

logfile = None

def sphere_top(y, x):
    z2 = rmax**2 - (x-x0)**2 - (y-y0)**2
    if z2 < 0:
        return 0
    else:
        return np.sqrt(z2)

def sphere_bottom(y, x):
    return -sphere_top(y,x)

def f(y, x, zdown, zup):
    top = min(sphere_top(y,x), zup)
    bottom = min(top,max(sphere_bottom(y,x), zdown))
    return top-bottom

def setup(kernel_language='Fortran', solver_type='classic', use_petsc=True,
          dimensional_split=False, outdir='_output', output_format='petsc',
          disable_output=True, num_cells=(64,64,64),
          tfinal=2.0, num_output_times=10, problem='heterogeneous'):
    """
    Example python script for solving the 3d acoustics equations.
    """
    from clawpack import riemann

    if use_petsc:
        import clawpack.petclaw as pyclaw
    else:
        from clawpack import pyclaw

    if solver_type=='classic':
        solver=pyclaw.ClawSolver3D(riemann.vc_acoustics_3D)
        solver.limiters = pyclaw.limiters.tvd.MC
    elif solver_type=='sharpclaw':
        solver = pyclaw.SharpClawSolver3D(riemann.vc_acoustics_3D)

    else:
        raise Exception('Unrecognized solver_type.')


    solver.bc_lower[0]=pyclaw.BC.periodic
    solver.bc_upper[0]=pyclaw.BC.periodic
    solver.bc_lower[1]=pyclaw.BC.periodic
    solver.bc_upper[1]=pyclaw.BC.periodic
    solver.bc_lower[2]=pyclaw.BC.periodic
    solver.bc_upper[2]=pyclaw.BC.periodic

    solver.aux_bc_lower[0]=pyclaw.BC.periodic
    solver.aux_bc_upper[0]=pyclaw.BC.periodic
    solver.aux_bc_lower[1]=pyclaw.BC.periodic
    solver.aux_bc_upper[1]=pyclaw.BC.periodic
    solver.aux_bc_lower[2]=pyclaw.BC.periodic
    solver.aux_bc_upper[2]=pyclaw.BC.periodic

    zl = 1.0  # Impedance in left half
    cl = 1.0  # Sound speed in left half

    if problem == 'homogeneous':
        if solver_type=='classic':
            solver.dimensional_split=True
        else:
            solver.lim_type = 1

        solver.limiters = [4]

        zr = 1.0  # Impedance in right half
        cr = 1.0  # Sound speed in right half

    if problem == 'heterogeneous':
        if solver_type=='classic':
            solver.dimensional_split=False

        solver.bc_lower[0]    =pyclaw.BC.wall
        solver.bc_lower[1]    =pyclaw.BC.wall
        solver.bc_lower[2]    =pyclaw.BC.wall
        solver.aux_bc_lower[0]=pyclaw.BC.wall
        solver.aux_bc_lower[1]=pyclaw.BC.wall
        solver.aux_bc_lower[2]=pyclaw.BC.wall

        zr = 2.0  # Impedance in right half
        cr = 2.0  # Sound speed in right half

    solver.limiters = pyclaw.limiters.tvd.MC

    # Initialize domain
    x = pyclaw.Dimension(-1.0, 1.0, num_cells[0], name='x')
    y = pyclaw.Dimension(-1.0, 1.0, num_cells[1], name='y')
    z = pyclaw.Dimension(-1.0, 1.0, num_cells[2], name='z')
    domain = pyclaw.Domain([x,y,z])

    num_eqn = 4
    num_aux = 2 # density, sound speed
    state = pyclaw.State(domain,num_eqn,num_aux)

    X,Y,Z = state.grid.p_centers

    state.aux[0,:,:,:] = zl*(X<0.) + zr*(X>=0.) # Impedance
    state.aux[1,:,:,:] = cl*(X<0.) + cr*(X>=0.) # Sound speed

    # Set initial density
    x0 = -0.5; y0 = 0.; z0 = 0.
    if problem == 'homogeneous':
        r = np.sqrt((X-x0)**2)
        width=0.2
        state.q[0,:,:,:] = (np.abs(r)<=width)*(1.+np.cos(np.pi*(r)/width))
    elif problem == 'heterogeneous':
        r = np.sqrt((X-x0)**2 + (Y-y0)**2 + (Z-z0)**2)
        width=0.1
        state.q[0,:,:,:] = (np.abs(r-0.3)<=width)*(1.+np.cos(np.pi*(r-0.3)/width))
    else:
        raise Exception('Unrecognized problem name')

    # Set initial velocities to zero
    state.q[1,:,:,:] = 0.
    state.q[2,:,:,:] = 0.
    state.q[3,:,:,:] = 0.

    tic1 = MPI.Wtime()
    claw = pyclaw.Controller()
    claw.solution = pyclaw.Solution(state, domain)
    claw.solver = solver
    claw.output_format = output_format
    claw.keep_copy = False
    if disable_output:
        claw.output_format = None
    claw.tfinal = tfinal
    claw.num_output_times = num_output_times
    claw.outdir = outdir + '_' + str(size)
    # Disable loggers
    #for i in claw.logger.handlers:
    #    i.setLevel(logging.CRITICAL)

    tic2 = MPI.Wtime()
    claw.run()
    toc = MPI.Wtime()
    timeVec1.array = toc - tic1
    timeVec2.array = toc - tic2
    t1 = MPI.Wtime()
    duration1 = timeVec1.max()[1]
    duration2 = timeVec2.max()[1]
    t2 = MPI.Wtime()

    if rank==0:
        logfile.Write(solver_type+' \n')
        logfile.Write('clawrun + load took '+str(duration1)+' seconds, for process '+str(rank)+'\n')
        logfile.Write('clawrun took '+str(duration2)+' seconds, for process '+str(rank)+' in grid '+str(mx)+'\n')
        logfile.Write('number of steps: '+ str(claw.solver.status.get('numsteps'))+'\n')
        logfile.Write('time reduction time '+str(t2-t1)+'\n')
        logfile.Write('tfinal '+str(claw.tfinal)+' and dt '+str(solver.dt)+'\n')
        logfile.Write('=' * 32 + '\n')

    return

if __name__=="__main__":
    tini = MPI.Wtime()
    import os
    import cProfile
    from scipy.special import cbrt
    import sys, getopt

    myopts, args = getopt.getopt(sys.argv[1:],"x:s:")
    mx=0
    tfinal_reduction = 1

    scaling = ''
    for o, a in myopts:
        if o == '-x':
            mx=a
        elif o == '-s':
            scaling = a
        else:
            print("Usage: %s -x cells -s scaling" % sys.argv[0])

    size = PETSc.Comm.getSize(PETSc.COMM_WORLD)
    rank = PETSc.Comm.getRank(PETSc.COMM_WORLD)

    solver = 'classic'

    if scaling=='weak':
        mx = int(96 * cbrt(size/4))
        tfinal_reduction = cbrt(size/4)

    # Initialize grid
    my = mz = mx
    out_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scaling_3d_{0}'.format(solver))
    processList = [0, 3]

    if rank==0 and not os.path.isdir(out_folder):
            os.mkdir(out_folder)

    # Initialize log file
    logfile = MPI.File.Open(MPI.COMM_WORLD, os.path.join(out_folder,"results_"+str(size)+'_'+str(mx)+".log"), MPI.MODE_CREATE|MPI.MODE_WRONLY)

    tb1call = MPI.Wtime()
    reduced_mx = cbrt(size/4) * 8
    setup(tfinal=2.0,num_cells=(reduced_mx,reduced_mx,reduced_mx),solver_type=solver,disable_output=True)
    PETSc.COMM_WORLD.barrier()

    tb2call = MPI.Wtime()    
    setup(tfinal=2.0,num_cells=(reduced_mx,reduced_mx,reduced_mx),solver_type=solver,disable_output=True)
    PETSc.COMM_WORLD.barrier()

    tb3call = MPI.Wtime()
    if rank in processList:
        funccall = "setup(tfinal=2.0/tfinal_reduction,num_cells=(mx,my,mz),solver_type=solver)"
        save_profile_name = os.path.join(out_folder,'statst_'+str(size)+'_'+str(mx)+"_"+str(rank))
        cProfile.run(funccall,save_profile_name)
    else:
        #print "process"+str(rank) +"not profiled"
        setup(tfinal=2.0/tfinal_reduction,num_cells=(mx,my,mz),solver_type=solver)

    PETSc.COMM_WORLD.barrier()
    tend = MPI.Wtime()
    timeVec3.array = tend - tini
    duration3 = timeVec3.max()[1]
    if MPI.COMM_WORLD.rank==0:
        logfile.Write('Total time : ' + str(duration3) + '\n')
        logfile.Write('Python dynamic loading: ' + str(duration3 - (2*(tb3call-tb2call)+ (tend- tb3call))) + '\n')