import os
import gzip
import datetime
import uuid

import zipfile as zf

from typing import Tuple


def define_file_type(filePath: str) -> Tuple(str, str):
    fileName, fineExtension = os.path.splitext(filePath)
    if fineExtension in ['.xes', '.csv', '.mxml']:
        fileName = fileName + fineExtension
        fineExtension = fineExtension
    elif fineExtension == '.gz':
        outfileName = fileName
        fileName, fineExtension = decompress_file_gzip(
            outfileName, filePath)
    elif fineExtension == '.zip':
        fileName, fineExtension = decompress_file_zip(
            fileName, filePath)
    else:
        raise IOError('file type not supported')
    return fileName, fineExtension


def decompress_file_gzip(outFileName: str, filePath: str) -> Tuple(str, str):
    '''
    Decompress .gz files
    '''
    inFile = gzip.open(filePath, 'rb')
    outFile = open(outFileName, 'wb')
    outFile.write(inFile.read())
    inFile.close()
    outFile.close()
    _, fileExtension = os.path.splitext(outFileName)
    return outFileName, fileExtension


def decompress_file_zip(outfilename: str, filePath: str) -> Tuple(str, str):
    '''
    Decompress .zip files
    '''
    with zf.ZipFile(filePath, "r") as zip_ref:
        zip_ref.extractall("../inputs/")
    _, fileExtension = os.path.splitext(outfilename)
    return outfilename, fileExtension


def folder_id():
    return (datetime.datetime.today()
            .strftime('%Y%m%d_') + str(uuid.uuid4())
            .upper()
            .replace('-', '_'))