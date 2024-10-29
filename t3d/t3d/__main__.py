from t3d.trinity_lib import TrinityEngine

import sys
import argparse
from importlib import metadata
# import time
from t3d.Logbook import info, log


def print_time(dt):
    ''' Print elapsed time '''
    h = int(dt // 3600)
    m = int((dt - 3600 * h) // 60)
    s = dt - 3600 * h - 60 * m
    info(f"Total time: {h:d}h {m:d}m {s:.1f}s")


def main():

    # Set-up command line arguments
    description = "Trinity3D profile prediction for stellarators and tokamaks"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("input_file",
                        help="Name of input file",
                        default=None)
    parser.add_argument("-l", "--log",
                        help="Specify the log file (optional)",
                        default=None)

    # Parse command line arguments
    args = parser.parse_args()

    # Look for input file
    try:
        fin = args.input_file
    except FileNotFoundError:
        sys.exit()

    if args.input_file[-3:] != '.in':
        print("T3D input file must have '.in' extension")
        sys.exit()

    # Create the log file
    if args.log:
        log_file = args.log
    else:
        log_file = args.input_file[:-3] + ".out"
    log.set_handlers(term_stream=True, file_stream=True, file_handler=log_file)

    info("Running T3D calculation.")
    info(f"Version = {metadata.version('t3d')}")

    info("\n  Loading input file: " + fin)
    info("  Logging output file: " + log_file)

    # start_time = time.time()

    # Start t3d
    engine = TrinityEngine(fin)
    engine.evolve_profiles()

    # end_time = time.time()
    # delta_t = end_time - start_time

    info('\nT3D Complete, exiting normally.')

    log.finalize()


if __name__ == "__main__":

    main()
