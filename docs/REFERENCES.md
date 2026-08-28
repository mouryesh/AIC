# References

The methods this project builds on, and how we relate to each. Every claim in
`METHOD.md` about prior work traces to something here.

---

## The mechanism we build on

**Li, L., Chang, Q., Ni, J. (2009). "Data driven bottleneck detection of
manufacturing systems."** *International Journal of Production Research*, 47(18).
<https://www.tandfonline.com/doi/full/10.1080/00207540701881860>

The **Turning Point Method**. Identifies the bottleneck as the station where the
trend changes from blockage exceeding starvation to starvation exceeding
blockage. **This is the same physical signal RippleTwin uses.** We implement it
as baseline `B3_TurningPoint`, in the strengthened deviation-scored form, and
report where it wins and where it cannot answer at all.

**Roser, C., Nakano, M., Tanaka, M. (2001–2002). "A practical bottleneck
detection method" / "Shifting bottleneck detection."** *Winter Simulation
Conference*; *International Symposium on Scheduling*.
<https://www.allaboutlean.com/wp-content/uploads/2014/05/2001_Roser_WSC-A-Practical-Bottleneck-Detection-Method_Preprint.pdf>

The **Active Period Method**: the bottleneck is the process with the longest
uninterrupted active period. Its documented limitation — "a very high data
requirement", needing to know precisely when every process is active — is the
assumption RippleTwin relaxes.
<https://www.allaboutlean.com/active-period-method/>

---

## Where the field says the gaps are

**Roda, I., Macchi, M., Skoogh, A. et al. (2023). "Throughput bottleneck
detection in manufacturing: a systematic review of the literature on methods and
operationalization modes."** *Production & Manufacturing Research*, 11(1).
<https://research.chalmers.se/publication/538741/file/538741_Fulltext.pdf>

Reviews 14 detection methods, classified by whether they use queue states,
process states, or both. Turning Point is classified as a queue-state method;
Active Period as a process-state method.

Two statements we quote, both verified verbatim in the full text:

- §5.2.4: *"None of the existing literature provides real-world validation of
  methods."*
- §5.2.6: calls for a digital twin that can "automatically analyze the throughput
  bottlenecks from the real-time data sets, predict the expected dynamics using
  data science, examine the different scenarios of eliminating the bottlenecks,
  and prescribe actions", continuously evolving using real-time data "and shop
  floor engineers' feedback."

**Honesty note.** An automated summary of this paper told us it named "partial
sensor coverage" and "stations without instrumentation" as explicit research
gaps. We checked the full text: **the word "sensor" does not appear in it.** We
do not make that claim, and we mention the near-miss because it is exactly the
kind of citation that would not survive a subject-matter expert.

---

## Adjacent work we are not

**Virtual metrology / soft sensors** (semiconductor manufacturing). Infers a
hard-to-measure property — film thickness, etch depth — from process variables
recorded *at the same instrumented station*.
<https://iopscience.iop.org/article/10.1088/1361-6501/ab4b39>

RippleTwin differs in what is missing: the station has **no instrumentation at
all**, so there are no local process variables to regress from. The information
comes from the *topology* — the stations either side of it.

**BSTAN and graph-attention approaches to bottleneck detection.** Model
spatial-temporal station interactions with GNNs over line topology.

We deliberately did not take this route. The propagation we need can be derived
from buffer capacity rather than learned, and a derived model needs no training
data, is inspectable by a plant engineer, and cannot drift. Where a learned
model would be genuinely better is non-serial topology — see the limitations.

**Data-enabled models for assembly lines with limited sensor information**
(2023). Evaluates line performance with reduced signal availability using a
recursive analytical model.
<https://www.sciencedirect.com/science/article/pii/S2213846323001463>

Adjacent, but "limited information" there means fewer signal types at monitored
stations, not stations that emit nothing.

---

## How to check us

The strongest check is not to read this list, it is to run
`B3_TurningPoint` against RippleTwin yourself:

```bash
python -c "import sys;sys.path.insert(0,'src');\
from rippletwin.evaluation.experiments import run_experiment;run_experiment()"
```

`results/tables/flow_faults_hidden_source.csv` reports both, calibrated to the
same false-alarm rate.
