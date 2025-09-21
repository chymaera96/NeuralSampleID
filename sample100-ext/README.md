# Sample100-ext Dataset

## Introduction

Sample100-ext is an extended version of the Sample100 dataset (a.k.a. ) originally created by Van Balen et al. [1, 2]. This dataset contains annotations of samples used in commercially released hip-hop/rap recordings.

The original Sample100 dataset has been enhanced with:

- Precise fine-grained temporal annotations: start and end times for all sample occurrences (±250ms resolution)
- Time-stretching ratio estimates between reference and query tracks
- Instrumentation (stem) annotations for both reference and query material
- Additional metadata and expert comments on the samples
- Annotation of the names of the noise (dummy) tracks

Sample100-ext is designed for evaluation of Automatic Sample Identification (ASID) systems, providing more detailed ground truth for segment-wise evaluation.

## Dataset Structure

The dataset consists of:

- 106 sample relationships between 76 query tracks (songs containing samples) and 68 reference tracks (original songs that were sampled)
- 137 sample occurrences (as some queries use multiple samples and some references appear in multiple queries)
- 320 additional "noise" tracks with similar genre distribution (not sampled in any query) to use as dummy tracks

The main CSV file (`samples.csv`) contains the following information:

- `sample_id`: Unique identifier for each sample relationship (e.g., "S001")
- `target_track_id` (previously `original_track_id`): ID of the reference track that was sampled (e.g., "T002")
- `query_track_id` (previously `sample_track_id`): ID of the query track containing the sample (e.g., "T001")
- `estimated_tempo_ratio`: Ratio of tempos (time-stretching factor) between query and reference tracks
- `target_instruments`: Instrumentation in the original sampled material
- `query_interfering_instruments`: Instruments overlaid on the sample in the query track
- `sample_type`: Categorization as "beat," "riff," or "1-note"
- `interpolation`: Whether the sample is direct ("no") or re-recorded ("yes")
- `comments`: Extra annotations about the sample

JSON files in the `annotations` folder (e.g., `S001.json`) provide detailed temporal annotations with precise start and end times for each sample occurrence in both the query and reference tracks.

The `sonic_visualiser_annotations` folder contains the originally annotated Sonic Visualiser files for each sample relationship, allowing for visual inspection of the annotations. The JSON files are generated from these Sonic Visualiser files.

The extra_annotations.json file contains additional comments by the annotators which may be useful for understanding the annotations, the samples and their context.

More information about the dataset, its class balance, and precisions about its creation can be found in [1,2,3].

## Usage

This dataset is intended for training and evaluating ASID systems. The precise temporal annotations allow for segment-wise evaluation, making it possible to test systems on short query segments.

Typical evaluation scenarios include:

1. Full-track retrieval: Using entire query tracks to retrieve corresponding reference tracks
2. Segment-wise retrieval: Using short segments from query tracks to evaluate system robustness
3. Sample transformation analysis: Evaluating performance across different types of sampling transformations

## Methodology

The extended annotations were created by expert musicians. The annotation process involved:

1. Precise temporal marking of all sample occurrences (±250ms resolution) by listening to the query and target tracks and annotating with Sonic Visualiser [4]
2. Calculation of time-stretching ratios using automatic beat tracking BeatThis [5] with manual verification
3. Annotation of instrumentation by listening to both original tracks and their source-separated [6] stems
4. Expert assessment of sample types, transformations, and other characteristics

## References

[1] Van Balen, J. (2011). Automatic Recognition of Samples in Musical Audio. Master Thesis, Universitat Pompeu Fabra, Barcelona, Spain.

[2] Van Balen, J., Serra, J., & Haro, M. (2012). Automatic Identification of Samples in Hip Hop Music. In Int. Symp. on Computer Music Modeling and Retrieval (CMMR). London, United Kingdom.

[3] THIS PAPER. Refining Music Sample Identification with a Self-Supervised Graph Neural Network. [Author(s) to be added later]

[4] Cannam, C., Landone, C., & Sandler, M. (2010). Sonic visualiser: An open source application for viewing, analysing, and annotating music audio files. In Proceedings of the ACM Multimedia 2010 International Conference, Firenze, Italy, October 2010, pp. 1467-1468.

[5] Foscarin, F., Schluter, J., & Widmer, G. (2024). Beat this! Accurate beat tracking without DBN postprocessing. In Proceedings of the 25th International Society for Music Information Retrieval Conference (ISMIR). San Francisco, CA, United States.

[6] Rouard, S., Massa, F., & D{\'e}fossez, A. (2023). Hybrid Transformers for Music Source Separation. In Proceedings of the 48th International Conference on Acoustics, Speech, and Signal Processing (ICASSP). IEEE.

<!-- ## License -->

## Citation

If you use this dataset in your research, please cite our paper:

```
[Citation information]
```
