# PTU_parser.py	Read PicoQuant Unified Histogram Files
# This is demo code. Use at your own risk. No warranties.
# Keno Goertz, PicoQUant GmbH, February 2018

# Note that marker events have a lower time resolution and may therefore appear 
# in the file slightly out of order with respect to regular (photon) event records.
# This is by design. Markers are designed only for relatively coarse 
# synchronization requirements such as image scanning. 

# T Mode data are written to an output file [filename]
# We do not keep it in memory because of the huge amout of memory
# this would take in case of large files. Of course you can change this, 
# e.g. if your files are not too big. 
# Otherwise it is best process the data on the fly and keep only the results.

import time
import sys
import struct
import numpy as np
import io
import os
import math

class PTU(object):
    def __init__(self):
        self.trace = []
        self.trace_binned = {'time':[], 'count':[]}
        self.trace_binned_gcd = {'time':[], 'count':[]}
        self.file_name = []
        self.file_name_wo_ext = []
        self.xlim = (0, -1)
        self.bin_size = 1e-3
        self.bin_gcd = 1e-3
        self.globRes = 0
        self.version = 0

    def read_ptu(self, file_name):
        # Tag Types
        tyEmpty8		= struct.unpack(">i", bytes.fromhex("FFFF0008"))[0]
        tyBool8		 = struct.unpack(">i", bytes.fromhex("00000008"))[0]
        tyInt8		= struct.unpack(">i", bytes.fromhex("10000008"))[0]
        tyBitSet64	= struct.unpack(">i", bytes.fromhex("11000008"))[0]
        tyColor8		= struct.unpack(">i", bytes.fromhex("12000008"))[0]
        tyFloat8		= struct.unpack(">i", bytes.fromhex("20000008"))[0]
        tyTDateTime	 = struct.unpack(">i", bytes.fromhex("21000008"))[0]
        tyFloat8Array = struct.unpack(">i", bytes.fromhex("2001FFFF"))[0]
        tyAnsiString	= struct.unpack(">i", bytes.fromhex("4001FFFF"))[0]
        tyWideString	= struct.unpack(">i", bytes.fromhex("4002FFFF"))[0]
        tyBinaryBlob	= struct.unpack(">i", bytes.fromhex("FFFFFFFF"))[0]

        # Record types
        rtPicoHarpT3	 = struct.unpack(">i", bytes.fromhex('00010303'))[0]
        rtPicoHarpT2	 = struct.unpack(">i", bytes.fromhex('00010203'))[0]
        rtHydraHarpT3	= struct.unpack(">i", bytes.fromhex('00010304'))[0]
        rtHydraHarpT2	= struct.unpack(">i", bytes.fromhex('00010204'))[0]
        rtHydraHarp2T3	 = struct.unpack(">i", bytes.fromhex('01010304'))[0]
        rtHydraHarp2T2	 = struct.unpack(">i", bytes.fromhex('01010204'))[0]
        rtTimeHarp260NT3 = struct.unpack(">i", bytes.fromhex('00010305'))[0]
        rtTimeHarp260NT2 = struct.unpack(">i", bytes.fromhex('00010205'))[0]
        rtTimeHarp260PT3 = struct.unpack(">i", bytes.fromhex('00010306'))[0]
        rtTimeHarp260PT2 = struct.unpack(">i", bytes.fromhex('00010206'))[0]
        rtMultiHarpNT3	 = struct.unpack(">i", bytes.fromhex('00010307'))[0]
        rtMultiHarpNT2	 = struct.unpack(">i", bytes.fromhex('00010207'))[0]

        # global variables
        global inputfile
        global outputfile_header
        global outputfile_records
        global recNum
        global oflcorrection
        global truensync
        global dlen
        global isT2
        global globRes
        global numRecords
        global lastRec
        global recCounter


        inputfile = open(file_name, "rb")
        self.file_name = file_name
        self.file_name_wo_ext, ext = os.path.splitext(file_name)
        outputfile_header = open(os.path.join(self.file_name_wo_ext+'_header.txt'), "w")
        # The following is needed for support of wide strings

        # Check if inputfile is a valid PTU file
        # Python strings don't have terminating NULL characters, so they're stripped
        magic = inputfile.read(8).decode("utf-8").strip('\0')
        if magic != "PQTTTR":
            print("ERROR: Magic invalid, this is not a PTU file.")
            inputfile.close()
            outputfile_header.close()
            return -1

        version = inputfile.read(8).decode("utf-8").strip('\0')
        outputfile_header.write("Tag version: %s\n" % version)

        # Write the header data to outputfile and also save it in memory.
        # There's no do ... while in Python, so an if statement inside the while loop
        # breaks out of it
        tagDataList = []  # Contains tuples of (tagName, tagValue)
        while True:
            tagIdent = inputfile.read(32).decode("utf-8").strip('\0')
            tagIdx = struct.unpack("<i", inputfile.read(4))[0]
            tagTyp = struct.unpack("<i", inputfile.read(4))[0]
            if tagIdx > -1:
                evalName = tagIdent + '(' + str(tagIdx) + ')'
            else:
                evalName = tagIdent
            outputfile_header.write("\n%-40s" % evalName)
            if tagTyp == tyEmpty8:
                inputfile.read(8)
                outputfile_header.write("<empty Tag>")
                tagDataList.append((evalName, "<empty Tag>"))
            elif tagTyp == tyBool8:
                tagInt = struct.unpack("<q", inputfile.read(8))[0]
                if tagInt == 0:
                    outputfile_header.write("False")
                    tagDataList.append((evalName, "False"))
                else:
                    outputfile_header.write("True")
                    tagDataList.append((evalName, "True"))
            elif tagTyp == tyInt8:
                tagInt = struct.unpack("<q", inputfile.read(8))[0]
                outputfile_header.write("%d" % tagInt)
                tagDataList.append((evalName, tagInt))
            elif tagTyp == tyBitSet64:
                tagInt = struct.unpack("<q", inputfile.read(8))[0]
                outputfile_header.write("{0:#0{1}x}".format(tagInt,18))
                tagDataList.append((evalName, tagInt))
            elif tagTyp == tyColor8:
                tagInt = struct.unpack("<q", inputfile.read(8))[0]
                outputfile_header.write("{0:#0{1}x}".format(tagInt,18))
                tagDataList.append((evalName, tagInt))
            elif tagTyp == tyFloat8:
                tagFloat = struct.unpack("<d", inputfile.read(8))[0]
                outputfile_header.write("%-3E" % tagFloat)
                tagDataList.append((evalName, tagFloat))
            elif tagTyp == tyFloat8Array:
                tagInt = struct.unpack("<q", inputfile.read(8))[0]
                outputfile_header.write("<Float array with %d entries>" % tagInt/8)
                tagDataList.append((evalName, tagInt))
            elif tagTyp == tyTDateTime:
                tagFloat = struct.unpack("<d", inputfile.read(8))[0]
                tagTime = int((tagFloat - 25569) * 86400)
                tagTime = time.gmtime(tagTime)
                outputfile_header.write(time.strftime("%a %b %d %H:%M:%S %Y", tagTime))
                tagDataList.append((evalName, tagTime))
            elif tagTyp == tyAnsiString:
                tagInt = struct.unpack("<q", inputfile.read(8))[0]
                tagString = inputfile.read(tagInt).decode("utf-8").strip("\0")
                outputfile_header.write("%s" % tagString)
                tagDataList.append((evalName, tagString))
            elif tagTyp == tyWideString:
                tagInt = struct.unpack("<q", inputfile.read(8))[0]
                tagString = inputfile.read(tagInt).decode("utf-16le", errors="ignore").strip("\0")
                outputfile_header.write(tagString)
                tagDataList.append((evalName, tagString))
            elif tagTyp == tyBinaryBlob:
                tagInt = struct.unpack("<q", inputfile.read(8))[0]
                outputfile_header.write("<Binary blob with %d bytes>" % tagInt)
                tagDataList.append((evalName, tagInt))
            else:
                print("ERROR: Unknown tag type")
                return -1
            if tagIdent == "Header_End":
                break

        # Reformat the saved data for easier access
        tagNames = [tagDataList[i][0] for i in range(0, len(tagDataList))]
        tagValues = [tagDataList[i][1] for i in range(0, len(tagDataList))]

        # get important variables from headers
        numRecords = tagValues[tagNames.index("TTResult_NumberOfRecords")]
        globRes = tagValues[tagNames.index("MeasDesc_GlobalResolution")]
        stopAfter = tagValues[tagNames.index("TTResult_StopAfter")]/1e3
        self.version = version
        self.globRes = globRes
        lastRec = 0
        recCounter = 0
        print("Version: %s, GlobalResolution: %e, StopAfter[s]: %f" %(version,globRes, stopAfter))
        print("Decoding %d records, this may take a while..." % numRecords)

        def gotOverflow(self, count):
            global outputfile_records, recNum

        def gotMarker(self, timeTag, markers):
            global outputfile_records, recNum
            outputfile_records.write("%u MAR_%2x %u\n" % (recNum, markers, timeTag))

        def gotPhoton(self, timeTag, channel, dtime):
            global outputfile_records, isT2, recNum, lastRec, recCounter
            if isT2:
                 return
            else:
                outputfile_records.write("%u CHN_%1x %u %8.0lf %10u\n" % (recNum, channel,\
                                 timeTag, (timeTag * globRes * 1e9), dtime))

        def readPT3(self):
            global inputfile, outputfile_records, recNum, oflcorrection, dlen, numRecords
            T3WRAPAROUND = 65536
            for recNum in range(0, numRecords):
                # The data is stored in 32 bits that need to be divided into smaller
                # groups of bits, with each group of bits representing a different
                # variable. In this case, channel, dtime and nsync. This can easily be
                # achieved by converting the 32 bits to a string, dividing the groups
                # with simple array slicing, and then converting back into the integers.
                try:
                    recordData = "{0:0{1}b}".format(struct.unpack("<I", inputfile.read(4))[0], 32)
                except:
                    print("The file ended earlier than expected, at record %d/%d."\
                        % (recNum, numRecords))
                    return -1

                channel = int(recordData[0:4], base=2)
                dtime = int(recordData[4:16], base=2)
                nsync = int(recordData[16:32], base=2)
                if channel == 0xF: # Special record
                    if dtime == 0: # Not a marker, so overflow
                        gotOverflow(1)
                        oflcorrection += T3WRAPAROUND
                    else:
                        truensync = oflcorrection + nsync
                        gotMarker(truensync, dtime)
                else:
                    if channel == 0 or channel > 4: # Should not occur
                        print("Illegal Channel: #%1d %1u" % (dlen, channel))
                    truensync = oflcorrection + nsync
                    gotPhoton(truensync, channel, dtime)
                    dlen += 1
                if recNum % 100000 == 0:
                    return

        def readPT2(self):
            global inputfile, outputfile_records, recNum, oflcorrection, numRecords
            T2WRAPAROUND = 210698240
            for recNum in range(0, numRecords):
                try:
                    recordData = "{0:0{1}b}".format(struct.unpack("<I", inputfile.read(4))[0], 32)
                except:
                    print("The file ended earlier than expected, at record %d/%d."\
                        % (recNum, numRecords))
                    return -1

                channel = int(recordData[0:4], base=2)
                time = int(recordData[4:32], base=2)
                if channel == 0xF: # Special record
                    # lower 4 bits of time are marker bits
                    markers = int(recordData[28:32], base=2)
                    if markers == 0: # Not a marker, so overflow
                        gotOverflow(1)
                        oflcorrection += T2WRAPAROUND
                    else:
                        # Actually, the lower 4 bits for the time aren't valid because
                        # they belong to the marker. But the error caused by them is
                        # so small that we can just ignore it.
                        truetime = oflcorrection + time
                        gotMarker(truetime, markers)
                else:
                    if channel > 4: # Should not occur
                        print("Illegal Channel: #%1d %1u" % (recNum, channel))
                        outputfile_records.write("\nIllegal channel ")
                    truetime = oflcorrection + time
                if recNum % 100000 == 0:
                    return

        def readHT3(self, version):
            global inputfile, outputfile_records, recNum, oflcorrection, numRecords
            T3WRAPAROUND = 1024
            for recNum in range(0, numRecords):
                try:
                    recordData = "{0:0{1}b}".format(struct.unpack("<I", inputfile.read(4))[0], 32)
                except:
                    print("The file ended earlier than expected, at record %d/%d."\
                         % (recNum, numRecords))
                    return -1

                special = int(recordData[0:1], base=2)
                channel = int(recordData[1:7], base=2)
                dtime = int(recordData[7:22], base=2)
                nsync = int(recordData[22:32], base=2)
                if special == 1:
                    if channel == 0x3F: # Overflow
                        # Number of overflows in nsync. If 0 or old version, it's an
                        # old style single overflow
                        if nsync == 0 or version == 1:
                            oflcorrection += T3WRAPAROUND
                            gotOverflow(1)
                        else:
                            oflcorrection += T3WRAPAROUND * nsync
                            gotOverflow(nsync)
                    if channel >= 1 and channel <= 15: # markers
                        truensync = oflcorrection + nsync
                        gotMarker(truensync, channel)
                else: # regular input channel
                    truensync = oflcorrection + nsync
                if recNum % 100000 == 0:
                    sys.stdout.write("\rProgress: %.1f%%" % (float(recNum)*100/float(numRecords)))
                    sys.stdout.flush()

        def readHT2(self, version):
            global inputfile, outputfile_records, recNum, oflcorrection, numRecords
            T2WRAPAROUND_V1 = 33552000
            T2WRAPAROUND_V2 = 33554432
            for recNum in range(0, 100):
                try:
                    recordData = "{0:0{1}b}".format(struct.unpack("<I", inputfile.read(4))[0], 32)
                    print(recordData)
                except:
                    print("The file ended earlier than expected, at record %d/%d."\
                         % (recNum, numRecords))
                    return -1

                special = int(recordData[0:1], base=2)
                channel = int(recordData[1:7], base=2)
                timetag = int(recordData[7:32], base=2)
                print(special, channel, timetag)
                if special == 1:
                    if channel == 0x3F: # Overflow
                        # Number of overflows in nsync. If old version, it's an
                        # old style single overflow
                        if version == 1:
                            oflcorrection += T2WRAPAROUND_V1
                            gotOverflow(1)
                        else:
                            if timetag == 0: # old style overflow, shouldn't happen
                                oflcorrection += T2WRAPAROUND_V2
                                gotOverflow(1)
                            else:
                                oflcorrection += T2WRAPAROUND_V2 * timetag
                    if channel >= 1 and channel <= 15: # markers
                        truetime = oflcorrection + timetag
                        gotMarker(truetime, channel)
                    if channel == 0: # sync
                        truetime = oflcorrection + timetag
                        gotPhoton(truetime, 0, 0)
                else: # regular input channel
                    truetime = oflcorrection + timetag
                    gotPhoton(truetime, channel+1, 0)
                if recNum % 100000 == 0:
                    sys.stdout.write("\rProgress: %.1f%%" % (float(recNum)*100/float(numRecords)))
                    sys.stdout.flush()

        oflcorrection = 0
        dlen = 0


        recordType = tagValues[tagNames.index("TTResultFormat_TTTRRecType")]
        if recordType == rtPicoHarpT2:
            isT2 = True
            print("PicoHarp T2 data")
            readPT2()
        elif recordType == rtPicoHarpT3:
            isT2 = False
            print("PicoHarp T3 data")
            readPT3()
        elif recordType == rtHydraHarpT2:
            isT2 = True
            print("HydraHarp V1 T2 data")
            readHT2(1)
        elif recordType == rtHydraHarpT3:
            isT2 = False
            print("HydraHarp V1 T3 data")
            readHT3(1)
        elif recordType == rtHydraHarp2T2:
            isT2 = True
            print("HydraHarp V2 T2 data")
            readHT2(2)
        elif recordType == rtHydraHarp2T3:
            isT2 = False
            print("HydraHarp V2 T3 data")
            readHT3(2)
        elif recordType == rtTimeHarp260NT3:
            isT2 = False
            print("TimeHarp260N T3 data")
            readHT3(2)
        elif recordType == rtTimeHarp260NT2:
            isT2 = True
            print("TimeHarp260N T2 data")
            with open(os.path.join(self.file_name_wo_ext+'_cache.dat'),'wb') as f:
                for b in inputfile:
                    f.write(b)
        elif recordType == rtTimeHarp260PT3:
            isT2 = False
            print("TimeHarp260P T3 data")
            readHT3(2)
        elif recordType == rtTimeHarp260PT2:
            isT2 = True
            print("TimeHarp260P T2 data")
            readHT2(2)
        elif recordType == rtMultiHarpNT3:
            isT2 = False
            print("MultiHarp150N T3 data")
            readHT3(2)
        elif recordType == rtMultiHarpNT2:
            isT2 = True
            print("MultiHarp150N T2 data")
            readHT2(2)
        else:
            print("ERROR: Unknown record type")
            return -1

        inputfile.close()
        outputfile_header.close()
        sys.stdout.flush()
        print("Finished")


    def processHT2(self, record, version=1, globRes=250e-12):
        T2WRAPAROUND_V1 = 33552000
        T2WRAPAROUND_V2 = 33554432

        if (version == 1):
            overflowCorrection = np.zeros(record['timetag'].shape)
            overflowCorrection[np.where(record['channel'] == 0x3F)[0]] = T2WRAPAROUND_V1
            overflowCorrection = np.cumsum(overflowCorrection)
        else:
            overflowCorrection = np.zeros(record['timetag'].shape, dtype=np.uint64)
            overflowCorrection[np.where((record['channel'] == 0x3F) & (record['timetag'] == 0))[0]] = T2WRAPAROUND_V2
            _loc = np.where((record['channel'] == 0x3F) & (record['timetag'] != 0))[0]
            overflowCorrection[_loc] = np.multiply(record['timetag'][_loc], T2WRAPAROUND_V2)
            overflowCorrection = np.cumsum(overflowCorrection)
            _tmp0 = np.add(record['timetag'],overflowCorrection)
            _tmp1 = _tmp0[np.where((record['special'] != 1) | (record['channel'] == 0x00))[0]].astype(np.uint64)
            record['truetime/ps'] = np.multiply(_tmp1, (globRes*1e12)).astype(np.uint64)
        return record


    def bin(self):
        _bin_size = np.uint64(self.bin_size)
        if ((self.xlim[0] >= 0) & (self.xlim[1] == -1)):
            _xlim = (np.uint64(np.ceil(self.xlim[0]*1e12/_bin_size)*_bin_size), np.uint64(self.trace['truetime/ps'][-1]))
        elif ((self.xlim[0] >= 0) & (self.xlim[1] > self.xlim[0])):
            _xlim = (np.uint64(np.ceil(self.xlim[0]*1e12/_bin_size)*_bin_size), np.uint64(np.floor(self.xlim[1]*1e12/_bin_size)*_bin_size))
        else:
            _xlim = (np.uint64(self.trace['truetime/ps'][0]), np.uint64(self.trace['truetime/ps'][-1]))
        binned_sparse = {'time': [], 'count': []}
        self.trace_binned = {'time': [], 'count': []}
        binned_sparse['time'], binned_sparse['count'] = \
        np.unique(np.multiply(np.floor_divide(self.trace['truetime/ps'][np.where((self.trace['truetime/ps'] >= _xlim[0]) & (self.trace['truetime/ps'] <= _xlim[1]))], _bin_size) \
                                , _bin_size).astype(np.uint64), return_counts=True)
#         print(_xlim, _bin_gcd)
        self.trace_binned['time'] = np.arange(_xlim[0], _xlim[1]+_bin_size, _bin_size, dtype=np.uint64)
        self.trace_binned['count'] = np.zeros(self.trace_binned['time'].shape)
        self.trace_binned['count'][np.nonzero(np.isin(self.trace_binned['time'], binned_sparse['time']))] = binned_sparse['count']

    def bin_else(self):
        _bin_size = np.uint64(self.bin_size)
        _bin_div = np.uint64(self.bin_size/self.bin_gcd)
        self.trace_binned = {'time': [], 'count': []}
        if (_bin_div == 1):
            self.trace_binned['time'] = self.trace_binned_gcd['time']
            self.trace_binned['count'] = self.trace_binned_gcd['count']
        else:
            self.trace_binned['time'], self.trace_binned['count'] = \
            self.trace_binned_gcd['time'][::_bin_div], \
            np.add.reduceat(self.trace_binned_gcd['count'], range(0,len(self.trace_binned_gcd['count']),_bin_div), dtype=np.uint64)

    def load_file(self, filename):
        self.read_ptu(self, filename)
        records = np.fromfile(os.path.splitext(filename)[0]+'_cache.dat', dtype=[('record', np.uint32)])
        self.trace = {'special':np.bitwise_and(np.right_shift(records['record'],31), 0x01).astype(np.byte),'channel':np.bitwise_and(np.right_shift(records['record'],25), 0x3F).astype(np.byte),'timetag':np.bitwise_and(records['record'], 0x1FFFFFF).astype(np.uint32)}
        self.trace_binned = {'time': [], 'count': []}
        return self.trace
