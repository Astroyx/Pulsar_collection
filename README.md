# Pulsar_collection

This is a new Parkes transient database (PTD II) that contains 165,592 single pulses from 363 known pulsars. We have re-processed all single pulse candidates from the first four years (1997-2001) of the Parkes Multibeam receiver system observation.

To run get_pulsar.py script, you may need to install the following python package:

* sqlite3

* astropy

* lmfit


Use the

     python get_pulsar.py -db  Pulsar_fits_database_v1.db -fluence -j J1745-3040 
     
command to extract all the file segments and fluence fitting result of PSR J1745-3040 in the database.
