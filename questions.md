
# Questions

<span style="color:#00DDFF">
Note: some parts of the questions between Tor and me overlap,
so I would suggest reading all of them first.
</span>


---
### 1. Cycles

Are the Cycles end inclusive? e.g. if the Cycle is Angle_ID(1, 0) will this return [1, 2, ..., 49, 0] (50 values) or [1, 2, ..., 48, 49] (49 values) 
I'm assuming it returns 50 values.

Why are the cycles split into `[1, 2, ..., 49, 0]` and `[26, 27, ..., 24, 25]`
instead of `[0, 1, ..., 48, 49]` and `[25, 26, ..., 23, 24]` which seems **MUCH** more intuitive?

Does this actually matter? Calculation of Range1 is based on Cycle_IDs, 
which will have the same value regardless of its index within the cycle.

**Solution: It makes no difference if we treat the element with Cycle_ID 0 as the first or 
the last element in a cycle since they are only ever retrieved by their Cycle_ID.**<br>
Please confirm this is correct.


---
### 2. CT (initial and definite)

#### 2.1

> The definite clotting time is determined when the CA reaches 4 mm<br>
> From the time of the detection of a CA of 4 mm you have to determine the time when the CA was 2 mm (backwards from 4 mm).     

This statement led to some confusion, the part about going back to the initial CT doesn't seem to make a lot of sense.<br>
The idea that CT was the interval of the points in time where CA reaches 2 and 4 mm respectively came up.<br>
_e.g.: `CT = t_4 - t_2 | where CA(t_2) = 2 and CA(t_4) = 4 -- in case of multiple t_2/t_4 the smallest will be used`_

However under that assumption the definition of CFT doesn't work:
> The CFT is calculated as the time interval between the definite CT and the point in time when the CA reaches or exceeds 20 mm. 

But if CT is an interval itself this becomes invalid since you cannot calculate an interval between a point in time and an interval.


#### 2.2

In `2024-04-01 09.02.28 Ch.4 IN-test heparin 1 u.xml` the `CT` is given as `1712`. But that is the first point where the amplitude reaches 2mm.<br>
This doesn't match the definition of CT in the Word document or the CT in `2024-03-25 14.53.14 Ch.1 EX-test fibrinolysis.xml`

Question:<br>
`CT = t_4 - t_2 | where CA(t_2) = 2 and CA(t_4) = 4 -- in case of multiple t_2/t_4 the smallest will be used`<br>
`CT = t_4  | CA(t_4) = 4 -- in case of multiple t_4 the smallest will be used`

**Which of these is correct?<br>
Why are the values in the 2 xml files we recieved seemingly using 2 different formulas?**



---
### 3. Range != Range1 

Range:
> Numeric representation of the distance of the extreme positions during each cycle
> (i.e. the position of the test system when it stands still)

Range1:
> Range1 is the representation of the range calculated directly from the raw data of the instrument<br>
> Range = Median(Angle_ID(7-19)) - Median(Angle_ID(32-44))

_(I assume `Range` instead of `Range1` here is a spelling mistake)_

Just make sure this is intentional and I understood it correctly.<br>
Range uses extremes and Range1 uses medians. Or are they actually the same and `Range = Range1`?<br>
_(This only matters if `Range` is used for `CA` instead of `Range2` - see next question)_



---
### 4. CA: which range is used?

```
CA(t) = (iRange-Range(t)) / iRange * GCC * CCC * TCC
```
Which range is used for the calculation of CA?
The formula suggests the regular `Range` but that seems to make the smoothing steps in `Range1` and `Range2` unnecessary.<br>
I think `Range2` should be used here.



---
### 5. MCF incorrect

MCF in my data is not the highest data point. 
It returns `34.92` instead of `~40` (same values as in the xml)

> - When a CA of at least 20 mm is reached and the current CA is less than 0.5 mm 
> larger than the CA of the value 10 lines before 

Maybe this line refers to full cycles instead of overlapping ones?
I assume "lines" here ie equivalent to values. 
This condition is true in my case way before the actual maximum value is reached.<br>
*Why can you not just use the maximum value?* It seems to be like that in
[similar publications](https://ashpublications.org/hematology/article/2020/1/67/474316/TEG-talk-expanding-clinical-roles-for)
(these might have nothing to do with the use case at hand, I have no medical knowledge about that).

**Questions:**  
1. **Are `MCF` and the `finalized MCF` the same value? Or are they treated as two individual values?**
1. **If they are different values, which one is used for the calculation of `ML`?**<br>
`ML = (1 - lowest CA after finalization of MCF / *MCF) * 100`

1. **What is the point of the conditions for finalizing `MCF`? Could you not just the maximum `CA` instead?**


![Clot graph](https://ash.silverchair-cdn.com/ash/content_public/journal/hematology/2020/1/10.1182_hematology.2020000090/4/m_hem2020000090cf3.png?Expires=1722335575&Signature=Nke-~KyWi0c1Par2fFw2VRakugmC3xdtzSKX9czNn8Qn0b2MAt-~oHTA-DTsdkX6o3zBLVTUlrOCia8k4H0qbAQwdO2pe-GQ6DDCvBhUHV~ygDyuOSASn7hzHz9URL~LquDfLhBSXjV79MRsEHRv3N1f9nCsgH5SFRG~CFLX4bmsyFkxkGQm56YLqPAQRdO-YOexuZEOA1brMPbb5ensw5rdOFHqTww0-UnAV~jp9HDeTSgkl9IrCK4w~wNexeJqQHajJXy9IAFJXh1HYhSZzYhNQRvUiqzcaRjXrn4kpvW1-UyPzlxo1TGKSRNVJtYrhiI879pMhGQWNUHEWMxjVQ__&Key-Pair-Id=APKAIE5G5CRDK6RD3PGA)
![MCF and ML](https://asa2.silverchair-cdn.com/asa2/content_public/journal/anesthesiology/121/1/10.1097_aln.0000000000000229/2/m_20ff01.png?Expires=1722344351&Signature=MteY8LRp5vABdMly9W17jRyRUgxhfayO1wk835nQcYSv9POVFGzym1M~YS8YnziNggq5rmZNF-MJQewEpkILFADIqZ9tobPM58KMhw0ffyuZv72zcDPaAwwOYr5KknYdCr8~SvADLobYj9fxa1wETNBqSLYIcC2itFh8cm3lncZ5uNEf31c2P9P4sZ8T01yfR-CuDyD9l3cMDgaHgmHXGZcbR5AGdbABhv55UtYgm56eiA3pya0GY55lq4e6qdzu3fmLmeTovF3xI0aEPj3f0sASHz0M17h0saWVaG6vIXMrhQzxzAEosVIcr7f77ZZfg7YG7QKCN0k54DoX~gMiCg__&Key-Pair-Id=APKAIE5G5CRDK6RD3PGA)


I included my source-code here, but I don't think there is an error in my logic.

```python
import pandas as pd

def _calculate_mcf(self) -> tuple[int, float]:
    amplitudes: pd.Series = self.table["amplitude"]
    highest_ca: float = 0.0
    amplitude: float
    for idx, amplitude in enumerate(amplitudes):
        if (
            (
                # The MCF is finalized either (whichever scenario is achieved first)
                # •	When 3 consecutive values are lower than the highest CA recorded prior to these 3 values
                sum(amplitudes.iloc[idx : idx + 3] < highest_ca) == 3
                # •	When a CA of at least 20 mm is reached and the current CA is less than 0.5 mm larger
                #   than the CA of the value 10 lines before
                or (amplitude >= 20 and 0 < (amplitude - amplitudes.iloc[idx - 10]) < 0.5)
            )
            # mcf is only shown from the point of the definite CT (defined as the point where "amplitude > 4")
            # It doesn't really make sense to be able to finalize it before that point
            and amplitude > 4
        ):
            # MCF is finalized
            break

        if amplitude > highest_ca:
            highest_ca = amplitude

    # noinspection PyUnboundLocalVariable
    return idx, highest_ca
```


---

# Tor's questions:

`Alt-Daten` bezieht sich auf folgende Datei:
`2024-03-25 14.53.14 Ch.1 EX-test fibrinolysis.xml`



---
### 1. CT

In den Alt-Daten wird ein CA >= 4mm erst bei t=64,8 erreicht. Trotzdem ist die CT in den Alt-Daten mit 60s ausgegeben. Das ist auf Anhieb nicht nachvollziehbar. Mit der neuen Berechnung wird eine CA >= 4mm bei t=62,3s schon erreicht. Das liegt an den überlappenden Sequenzen.

Die folgende Aussage verstehe ich ehrlich gesagt nicht: "From the time of the detection of a CA of 4 mm you have to determine the time when the CA was 2 mm (backwards from 4 mm). "



---
### A5, A10, A20

Die Überschriften sind falsch, ist: "amplitude N mm after CT", soll: "amplitude N min after CT". In der Überschrift zu A20 steht auch "10 mm" statt "20 min".

In dem Text zu A20 steht CT+600s, es soll CT+1200s sein.

Es gibt hier in den Alt-Daten kleine Abweichungen von meinem händischen Vergleich. Vielleicht als Folgefehler von der Abweichung in der CT-Berechnung, vielleicht sind es aber nur Rundungsfehler.

ist: A5 = 29, soll: A5 = CA(364,8) = 29,65

ist: A10 = 37, soll: A10 = CA(664,8) = 36,88

ist: A20 = 39, soll: A20 = CA(1264,8) = 39,29

Wenn A10 in der Ausgabe kaufmännisch gerundet wird, hätte ich das auch bei A5 erwartet. Dort ist es aber nicht so.



---
### CFT

Hier gibt es in den Alt-Daten vielleicht die größte Abweichung zu der beschriebenen Formel. Eine CA >= 20mm wird in den Alt-Daten bei t=194,8s erreicht. Jetzt hängt die Weiterberechnung auch von der Abweichung in CT ab, aber wenn ich mit CT=60s rechne, kommt CFT=135s raus. Wenn ich mit CT=64,8s rechne, kommt CFT=130s raus. Ausgegeben in dem XML mit Alt-Daten ist aber CFT=145s.



---
### LOT und LT

Hier sind die Abweichungen vermutlich auch durch die überlappenden Sequenzen und der höheren zeitlichen Granularität geschuldet. Die Abweichungen sind bei beiden Werten 2,5s.



---
### ML30, ML45 und ML60

Hier finde ich keine Werte in den Alt-Daten, womit ich überhaupt vergleichen könnte.

