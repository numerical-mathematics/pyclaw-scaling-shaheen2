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
          dimensional_split=False, outdir='_output', output_format='hdf5',
          disable_output=True, num_cells=(64,64,64),
          tfinal=0.10, num_output_times=10):
    if use_petsc:
        import clawpack.petclaw as pyclaw
    else:
        from clawpack import pyclaw

    if use_petsc:
        import clawpack.petclaw as pyclaw
    else:
        from clawpack import pyclaw

    if solver_type=='classic':
        solver = pyclaw.ClawSolver3D(riemann.euler_3D)
        solver.dimensional_split = dimensional_split
        solver.limiters = pyclaw.limiters.tvd.minmod
        solver.cfl_max = 0.6
        solver.cfl_desired = 0.55
        solver.dt_initial = 3e-4
    else:
        raise Exception('Unrecognized solver_type.')
    
    size = PETSc.Comm.getSize(PETSc.COMM_WORLD)
    rank = PETSc.Comm.getRank(PETSc.COMM_WORLD) 
    
    x = pyclaw.Dimension(-1.0, 1.0, num_cells[0], name='x')
    y = pyclaw.Dimension(-1.0, 1.0, num_cells[1], name='y')
    z = pyclaw.Dimension(-1.0, 1.0, num_cells[2], name='z')
    domain = pyclaw.Domain([x,y,z])

    state = pyclaw.State(domain,num_eqn)

    state.problem_data['gamma']=gamma

    grid = state.grid
    X,Y,Z = grid.p_centers
    r = np.sqrt((X-x0)**2 + (Y-y0)**2 + (Z-z0)**2)

    state.q[density,   :,:,:] = 1.0
    state.q[x_momentum,:,:,:] = 0.
    state.q[y_momentum,:,:,:] = 0.
    state.q[z_momentum,:,:,:] = 0.

    background_pressure = 1.0e-2
    Eblast = 0.851072
    pressure_in = Eblast*(gamma-1.)/(4./3.*np.pi*rmax**3)
    state.q[energy,:,:,:] = background_pressure/(gamma-1.) # energy (e)

    # Compute cell fraction inside initial perturbed sphere
    dx, dy, dz = state.grid.delta
    dx2, dy2, dz2 = [d/2. for d in state.grid.delta]
    dmax = max(state.grid.delta)

    for i in xrange(state.q.shape[1]):
        for j in xrange(state.q.shape[2]):
            for k in xrange(state.q.shape[3]):
                if r[i,j,k] - dmax > rmax:
                    continue
                else:
                    p = pressure_in # pressure
                    state.q[energy,i,j,k] = p/(gamma-1.) # energy (e)


    solver.all_bcs = pyclaw.BC.extrap

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
    claw.outdir = outdir
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

    myopts, args = getopt.getopt(sys.argv[1:],"x:")
    mx=0
    tfinal_reduction = 1
    for o, a in myopts:
        if o == '-x':
            mx=a
        elif o == '-s':
            scaling = a
        else:
            print("Usage: %s -s scaling -x cells" % sys.argv[0])

    size = PETSc.Comm.getSize(PETSc.COMM_WORLD)
    rank = PETSc.Comm.getRank(PETSc.COMM_WORLD)

    solver = 'classic'

    if scaling=='weak':
        mx = int(96 * cbrt(size/4))
        tfinal_reduction = cbrt(size/4)

    # Initialize grid
    my = mz = mx
    print str((mx, my, mz))
    out_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scaling_3d_{0}'.format(solver))
    processList = [0, 3]

    if rank==0 and not os.path.isdir(out_folder):
            os.mkdir(out_folder)

    # Initialize log file
    logfile = MPI.File.Open(MPI.COMM_WORLD, os.path.join(out_folder,"results_"+str(size)+'_'+str(mx)+".log"), MPI.MODE_CREATE|MPI.MODE_WRONLY)

    tb1call = MPI.Wtime()
    reduced_mx = cbrt(size/4) * 8
    setup(tfinal=0.1,num_cells=(reduced_mx,reduced_mx,reduced_mx),solver_type=solver)
    PETSc.COMM_WORLD.barrier()

    tb2call = MPI.Wtime()    
    setup(tfinal=0.1,num_cells=(reduced_mx,reduced_mx,reduced_mx),solver_type=solver)
    PETSc.COMM_WORLD.barrier()

    tb3call = MPI.Wtime()
    if rank in processList:
        funccall = "setup(tfinal=0.1/tfinal_reduction,num_cells=(mx,my,mz),solver_type=solver)"
        save_profile_name = os.path.join(out_folder,'statst_'+str(size)+'_'+str(mx)+"_"+str(rank))
        cProfile.run(funccall,save_profile_name)
    else:
        #print "process"+str(rank) +"not profiled"
        setup(tfinal=0.1/tfinal_reduction,num_cells=(mx,my,mz),solver_type=solver)

    PETSc.COMM_WORLD.barrier()
    tend = MPI.Wtime()
    timeVec3.array = tend - tini
    duration3 = timeVec3.max()[1]
    if MPI.COMM_WORLD.rank==0:
        logfile.Write('Total time : ' + str(duration3) + '\n')
        logfile.Write('Python dynamic loading: ' + str(duration3 - (2*(tb3call-tb2call)+ (tend- tb3call))) + '\n')