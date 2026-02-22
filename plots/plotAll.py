import plots as plots
import os

#-----------------------------------------------------------
def main():
    grids = ['steadystate']
    orders = ['1st','2nd']
    types = ['roe','HLLE']
    grifile = 'grids/coarse_local_refinement_1.gri'
    for grid in grids:
        for order in orders:
            for type in types:
                print('Plotting ' + grid + ' ' + order + ' ' + type)
                solutionfile = 'build/' + grid + '.' + order + '.' + type + '.data.txt'
                errorfile = 'build/' + grid + '.' + order + '.' + type + '.residual.txt'
                c_xfile = 'build/' + grid + '.' + order + '.' + type + '.c_x.txt'
                c_yfile = 'build/' + grid + '.' + order + '.' + type + '.c_y.txt'
                outprefix = 'plots/' + grid + '_' + order + '_' + type

                os.makedirs(outprefix, exist_ok=True)

                plots.plotsolution(grifile,solutionfile,'mach',outprefix + '/mach.png')
                plots.plotsolution(grifile,solutionfile,'entropy',outprefix + '/entropy.png')
                plots.ploterror(errorfile,outprefix + '/error.png')
                plots.plotcoefficients(c_xfile,'x',outprefix + '/c_x.png')
                plots.plotcoefficients(c_yfile,'y',outprefix + '/c_y.png')
                plots.plotcp(grifile,solutionfile,outprefix + '/cp.png')

#-----------------------------------------------------------
if __name__ == "__main__":
    main()

