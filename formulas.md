
|     Name     | Description                                                                                                                                                                                                                       |
|:------------:|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|      *       | the “*” before a numeric parameter represents that the parameter may not be final, i.e. that it may change during the course of the measurement                                                                                   |
| **Angle_ID** | numeric representation of the angle position of the detection system of the Clot firmness analyzer                                                                                                                                |
|    **CA**    | Clot Amplitude (in mm)                                                                                                                                                                                                            |
|    **CT**    | Clotting time in sec                                                                                                                                                                                                              |
|  **Cycle**   | 50 consecutive datapoints starting either at Cycle_ID == 1 or 26 (every 25 values)                                                                                                                                                |
| **Cycle_ID** | Representation of the position of the detection system during the test cycle. A test cycle can begin at position 1 and end at position 00 or begin at position 26 and end at position 25. The time between 2 cycle IDs is 0.1 sec |
|  **Range**   | Numeric representation of the distance of the extreme positions during each cycles (i.e. the position of the test system when it stands still)                                                                                    |
|  **Range1**  | Range1 is the representation of the range calculated directly from the raw data of the instrument                                                                                                                                 |
|  **Range2**  | Range2 is the representation of the range calculated by smoothing Range1                                                                                                                                                          |
|   **Time**   | “Time” is the timestamp in sec for the actual data position in relation of the start of the test, which is set as t=0 sec                                                                                                         |
|   **(t)**    | (t) represents that we refer to a specific timestamp for the respective value.                                                                                                                                                    |


Formulas:

```
Angle_ID(Cycle_ID_start, Cycle_ID_end, t) = a set of Angle_IDs between the given Cycle_IDs starting from a given point in time (t). End inclusive
```

```
Range(t) = Max(Angle_ID(0, 49, t)) - Min(Angle_ID(0, 49, t)
```

```
Range1(t) = Median(Angle_ID(7-19, t)) - Median(Angle_ID(32-44, t))
```
_Range and Range1 are applied to each cycle, so they return a dataset 1/25th the size of the original data._

```
Range2(t) = Median([Range1(t - 3 * 2.5), Range1(t - 2 * 2.5), ..., Range1(t + 3 * 2.5)])
```

```
iRange = Median([Range2(3 * 2.5), ..., Range2(7 * 2.5)])
```
_*2.5 scales the index to the time position since cycles are 2.5s apart.
If the first cycle doesn't start at t=0 an offset has to be added._

```
CA(t) = ((iRange-Range(t)) / iRange * GCC * CCC * TCC) * 100
```
_I think Range2(t) should be used here instead._

```
ML = (1 - lowest CA after finalization of MCF / MCF) * 100
```
