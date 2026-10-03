# Ridgecrest foreshock fixture

The 2019-07-04 17:33:49 UTC Ridgecrest foreshock (Mw 6.4) as `examples/ridgecrest_obspy.ipynb` downloads it,
so the notebook runs offline. `ORIGIN_TIME = UTCDateTime("2019-07-04T17:33:49")`.

`events.xml.gz` (QuakeML) is the USGS event followed by the ISC event:

```python
usgs = Client("USGS").get_events(eventid="ci38443183", includeallorigins=True, includeallmagnitudes=True)
isc = Client("ISC").get_events(starttime=ORIGIN_TIME - 60, endtime=ORIGIN_TIME + 60, minmagnitude=6,
                               includeallorigins=True, includeallmagnitudes=True)
```

The USGS event carries the USGS Mww and Mwb tensors and the SCSN (CI) TMTS tensor; the ISC event carries
GCMT, GFZ and NEIC's copies of the USGS tensors. The ISC event's picks, arrivals, amplitudes and station
magnitudes are removed (they are most of its size and the notebook does not read them).

`stations.xml.gz` (StationXML with responses) and `waveforms.mseed` are, for each station, from its own data
centre over `ORIGIN_TIME - 1500` to `ORIGIN_TIME + 500`:

| data centre | stations | channels |
|---|---|---|
| `Client("SCEDC")` | CI.PDM, CI.SCI2, CI.SMR | `BH?` |
| `Client("NCEDC")` | BK.PACP, BK.WELL | `BH?`, location `00` |
| `Client("EARTHSCOPE")` | NN.Q09A, NN.SHP (HH only in this sector, broadband) and PY.BPH05 | `HH?`, `BH?` |

each fetched with one `get_waveforms(network, station, location, channel, start, end)` and one
`get_stations(..., level="response")` call per station (SCEDC returns no data to a bulk request for these
stations). The raw counts are then reduced to 1 Hz so the fixture stays small:
`stream.merge(method=0); stream.detrend("linear"); stream.resample(1.0)`, rounded to int32 and written as
STEIM2 MiniSEED. The two XML files are gzipped (`gzip -9`); ObsPy reads them as they are.
