import plots as plots
import os

#-----------------------------------------------------------
def main():
    grids = ['Steadystate']
    orders = ['1stOrder','2ndOrder']
    types = ['RoeFlux','HLLEFlux']
    grifile = 'grids/coarse_local_refinement_1.gri'
    for grid in grids:
        for order in orders:
            for type in types:
                print('Plotting ' + grid + ' ' + order + ' ' + type)
                solutionfile = 'build/' + grid + '.' + order + '.' + type + '.data.txt'
                errorfile = 'build/' + grid + '.' + order + '.' + type + '.residual.txt'
                c_xfile = 'build/' + grid + '.' + order + '.' + type + '.c_x.txt'
                c_yfile = 'build/' + grid + '.' + order + '.' + type + '.c_y.txt'
                outprefix = grid + '.' + order + '.' + type
                outfile = 'plots/' + outprefix

                os.makedirs(outfile, exist_ok=True)

                plots.plotsolution(grifile,solutionfile,'mach',outfile + '/' + outprefix + '.mach.png')
                plots.plotsolution(grifile,solutionfile,'entropy',outfile + '/' + outprefix + '.entropy.png')
                plots.ploterror(errorfile,outfile + '/' + outprefix + '.error.png')
                plots.plotcoefficients(c_xfile,'x',outfile + '/' + outprefix + '.c_x.png')
                plots.plotcoefficients(c_yfile,'y',outfile + '/' + outprefix + '.c_y.png')
                plots.plotcp(grifile,solutionfile,outfile + '/' + outprefix + '.cp.png')

#-----------------------------------------------------------
if __name__ == "__main__":
    main()
