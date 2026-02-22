import plots.plots as plots
import subprocess
import os
import glob
import re

EXECUTABLE = "./proj-2"
GRIFILE = "../grids/coarse_local_refinement_1.gri"
SNAME = "mach"
FRAMERATE  = 10

def run_simulation():
    print('Running simulation...')
    result = subprocess.run(
        [EXECUTABLE],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print('STDERR:', result.stderr)
        raise RuntimeError(f'Simulation failed with return code {result.returncode}')
    print(result.stdout)
    print('Simulation complete.')

def render_frames():
    os.makedirs('frames', exist_ok=True)
    files = glob.glob('Freestream.2ndOrder.RoeFlux.data.txt')
    #files.sort(key=lambda f: float(re.search(r'\.([0-9]+\.[0-9]+)\.data', f).group(1)))

    if not files:
        print('No unsteady data files found.')
        return

    print(f'Found {len(files)} frames to render...')
    for i, f in enumerate(files):
        outfile = f'frames/frame_{i:04d}.png'
        print(f'  [{i+1}/{len(files)}] {f} -> {outfile}')
        plots.plotsolution(GRIFILE, f, SNAME, outfile)

def make_movie():
    print('Running ffmpeg...')
    result = subprocess.run([
        'ffmpeg', '-y',
        '-framerate', str(FRAMERATE),
        '-i', 'frames/frame_%04d.png',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        'output.mp4'
    ], capture_output=True, text=True)
    if result.returncode != 0:
        print('ffmpeg STDERR:', result.stderr)
        raise RuntimeError('ffmpeg failed')
    print('Movie saved to output.mp4')

if __name__ == '__main__':
    #run_simulation()
    render_frames()
    #make_movie()
