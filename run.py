#! /usr/bin/env python3

from pytocl.protocol import Client
from ANNDriver import ANNDriver
from ANNDriver_jasper import ANNDriverJasper
import argparse
import logging
import os

if __name__ == '__main__':
    filepath = os.path.realpath(__file__)

    parser = argparse.ArgumentParser(
        description="")
    parser.add_argument(
        "-f", "--model_file", help="The path to the trained driver model",
        default=os.path.dirname(filepath) + "/models/NNdriver2-100-300.pt"
    )
    parser.add_argument(
        "-H", "--hidden", help="Set the number of hidden neurons",
        default="100", type=int
    )
    parser.add_argument(
        "-d", "--depth", help="Set the number of layers",
        default="2", type=int
    )
    parser.add_argument(
        "-n", "--norm", help="Normalize sensor values between 0 and 1",
        default=False, action='store_true'
    )
    parser.add_argument(
        "-dump", "--dump",
        default=False, type=bool
    )
    parser.add_argument(
        "-r", "--record", help="The path to a file that will contain recorded \
        actuator & sensor data", action="store_true", default=None
    )

    parser.add_argument(
        '--hostname',
        help='Racing server host name.',
        default='localhost'
    )
    parser.add_argument(
        '-p',
        '--port',
        help='Port to connect, 3001 - 3010 for clients 1 - 10.',
        type=int,
        default=3001
    )
    parser.add_argument('-v', help='Debug log level, 0 is no logging, 1 is info, 2 is debug.',
                        default='0', metavar="LVL", type=int)

    args = parser.parse_args()

    # switch log level:
    if args.v == 2:
        level = logging.DEBUG
    elif args.v == 1:
        level = logging.INFO
    else:
        # Suppress all logging output
        level = 100000

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)7s %(name)s %(message)s"
    )

    # Init client
    if args.dump:
        client = Client(
            driver=ANNDriverJasper(
                args.model_file, args.hidden, args.depth, args.port, args.record, args.norm),
            port=args.port)
    else:
        client = Client(
            driver=ANNDriver(
                args.model_file, args.hidden, args.depth, args.record, args.norm),
            port=args.port)

    try:
        # start client loop:
        client.run()
    except KeyboardInterrupt as e:
        print("Exit")
        exit()
