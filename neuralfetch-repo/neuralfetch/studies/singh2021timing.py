# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp

import mne
import pandas as pd
from mne_bids import BIDSPath

from neuralfetch import download
from neuralset.events import study

logger = logging.getLogger(__name__)


class Singh2021Timing(study.Study):
    url: tp.ClassVar[str] = (
        "https://huggingface.co/datasets/jalauer/Singh2021Timing/tree/main/data"
    )
    """Singh2021Timing: EEG responses during an interval timing task in Parkinson's disease.

    EEG recordings from 83 Parkinson's disease patients and 37 healthy controls
    during a peak-interval timing task with 3-second and 7-second target intervals.
    A subset of 9 PD patients completed sessions both ON and OFF dopaminergic
    medication.

    Experimental Design:
        - EEG recordings (63-channel, 500 Hz)
        - 120 participants (83 PD, 37 controls), 138 timelines
        - 80 trials per session (40 per interval type)
        - Paradigm: peak-interval timing task with visual distractors

    Notes:
        - Population includes Parkinson's disease patients and healthy controls.
        - 9 PD participants have two sessions (ON/OFF medication) recorded under
          different subject IDs (see _SUBJECT_ALIASES).
    """

    bibtex: tp.ClassVar[str] = """
    @article{singh2021timing,
        title={Timing Variability and Midfrontal \textasciitilde 4 {{Hz}} Rhythms Correlate with Cognition in {{Parkinson}}'s Disease},
        author={Singh, Arun and Cole, Rachel C. and Espinoza, Arturo I. and Evans, Aron and Cao, Scarlett and Cavanagh, James F. and Narayanan, Nandakumar S.},
        year=2021,
        month=feb,
        journal={npj Parkinson's Disease},
        volume={7},
        number={1},
        pages={14},
        publisher={Nature Publishing Group},
        issn={2373-8057},
        doi={10.1038/s41531-021-00158-x},
        copyright={2021 The Author(s)},
        langid={english},
        keywords={Neurophysiology,Neuroscience},
    }
        url = {https://huggingface.co/datasets/jalauer/Singh2021Timing/tree/main/data},

    @misc{singh2021_data,
        url={https://huggingface.co/datasets/jalauer/Singh2021Timing/tree/main/data}
    }
    """
    licence: tp.ClassVar[str] = "PDDL-1.0"
    description: tp.ClassVar[str] = (
        "EEG recordings in 83 Parkinson's disease patients and 37 controls during interval timing."
    )
    _SUBJECT_RUNS: tp.ClassVar[dict[str, tp.Iterable[int]]] = {
        "Control": set(range(1025, 1420, 10)) - set((1045, 1165, 1355)),
        "PD": (
            set(range(1005, 1870, 10))
            | set(
                [
                    2445,
                    2515,
                    2565,
                    2625,
                    2815,
                    2835,
                    2845,
                    2845,
                    2855,
                    2865,
                    3445,
                    3515,
                    3565,
                    3625,
                ]
            )
        )
        - set((1205, 1255, 1345, 1355, 1495, 1545, 1805, 1825)),
    }
    # There are 9 PD subjects with two sessions, but recorded under a different subject name in
    # session 2 (see README in url above)
    _SUBJECT_ALIASES: tp.ClassVar[dict[str, str]] = {
        "PD1815": "PD2815",
        "PD1835": "PD2835",
        "PD1845": "PD2845",
        "PD1855": "PD2855",
        "PD1865": "PD2865",
        "PD3445": "PD2445",
        "PD3515": "PD2515",
        "PD3565": "PD2565",
        "PD3625": "PD2625",
    }
    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=138,
        num_subjects=129,
        num_events_in_query=882,  # query=1st timeline
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(63, 733750),
        frequency=500,
    )

    def _download(self, overwrite: bool = False) -> None:
        # https://predict.cs.unm.edu/downloads.php d014
        # Unable to use original dataset on PRED+ct.
        # Files shared here rely on Sharepoint
        # and cannot be programmatically downloaded there.
        # Alternate source found on Hugging Face, not uploaded by original authors.
        # https://huggingface.co/jalauer/datasets
        hf_org = "jalauer"
        hf_repo = "Singh2021Timing"
        hg = download.Huggingface(org=hf_org, study=hf_repo, dset_dir=self.path)
        if hg.get_success_file().exists() and not overwrite:
            return
        hg.download(overwrite=overwrite)

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        """Returns a generator of all recordings"""
        for diagnosis, subjects in self._SUBJECT_RUNS.items():
            for subject in sorted(subjects):
                sub_id = f"{diagnosis}{subject}"
                sessions = [1, 2] if sub_id in self._SUBJECT_ALIASES else [1]
                for session in sessions:
                    yield dict(subject=sub_id, session=session, diagnosis=diagnosis)

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        tl = timeline
        subject = (
            self._SUBJECT_ALIASES.get(tl["subject"], tl["subject"])
            if tl["session"] == 2
            else tl["subject"]
        )
        basename = self.path / "download" / "data" / subject

        # extract annotations
        events = mne.read_annotations(basename.with_suffix(".vmrk")).to_data_frame(
            time_format=None
        )
        events.rename(columns={"onset": "start"}, inplace=True)
        events["type"] = "Stimulus"
        events["code"] = events.description.map(
            {
                "Stimulus/S  1": 1,
                "Stimulus/S  2": 2,
                "Stimulus/S  3": 3,
                "Stimulus/S  4": 4,
                "Stimulus/S  5": 5,
                "Stimulus/S  6": 6,
                "Stimulus/S  7": 7,
                "Stimulus/S255": 8,
                "New Segment/": 9,
                "Response/R  3": 10,
            }
        )
        events["description"] = events.description.map(
            {
                "Stimulus/S  1": "short_interval_instruction",  # 1 s
                "Stimulus/S  2": "long_inverval_instruction",  # 1 s
                "Stimulus/S  3": "interval_start",  # Start of interval (blue rectangle shown)
                # Lasts 8-10 s for short intervals and 18-20 s for long intervals
                "Stimulus/S  4": "spacebar_press",  # XXX Replace with Button event of right duration
                "Stimulus/S  5": "spacebar_release",
                "Stimulus/S  6": "distracting_vowel",
                # XXX Interval end is not available in the dataset
                "Stimulus/S  7": "trial_feedback",  # On 15% of the trials
                "Stimulus/S255": "end_of_last_trial",
                # Not described in README.md
                "New Segment/": "unknown",
                "Response/R  3": "unknown",
            }
        )
        events.loc[
            (
                (events.type == "Stimulus")
                & events.description.str.endswith("interval_instruction")
            ),
            "duration",
        ] = 1.0

        eeg = dict(type="Eeg", filepath=basename.with_suffix(".vhdr"), start=0)
        events = pd.concat([pd.DataFrame([eeg]), events], ignore_index=True)
        return events


class Singh2021Timing_Openneuro(study.Study):
    url: tp.ClassVar[str] = "https://openneuro.org/datasets/ds004579/versions/1.0.0"
    """Singh2021Timing_Openneuro: OpenNeuro (ds004579) version of the interval timing task.

    Workshop copy of :class:`Singh2021Timing` sourced from OpenNeuro instead of
    Hugging Face, so the two can coexist while the OpenNeuro pipeline is validated.
    Provided in BIDS-compliant format with EEGLAB ``.set``/``.fdt`` recordings and
    BIDS ``_events.tsv`` sidecars.

    Experimental Design:
        - EEG recordings (63-channel, 500 Hz)
        - 139 recordings (94 Parkinson's disease, 45 healthy controls)
        - Paradigm: peak-interval timing task with visual distractors

    Notes:
        - The original diagnosis/subject label (e.g. ``PD1005``, ``Control1025``)
          is available in ``participants.tsv`` under the ``EEG`` column; BIDS uses
          numeric ``sub-XXX`` identifiers.
        - Single-session only: the 9 PD dual-session recordings aliased in the
          Hugging Face version are not part of ds004579.
    """

    bibtex: tp.ClassVar[str] = """
    @article{singh2021timing,
        title={Timing Variability and Midfrontal \textasciitilde 4 {{Hz}} Rhythms Correlate with Cognition in {{Parkinson}}'s Disease},
        author={Singh, Arun and Cole, Rachel C. and Espinoza, Arturo I. and Evans, Aron and Cao, Scarlett and Cavanagh, James F. and Narayanan, Nandakumar S.},
        year=2021,
        month=feb,
        journal={npj Parkinson's Disease},
        volume={7},
        number={1},
        pages={14},
        publisher={Nature Publishing Group},
        issn={2373-8057},
        doi={10.1038/s41531-021-00158-x},
        copyright={2021 The Author(s)},
        langid={english},
        keywords={Neurophysiology,Neuroscience},
    }

    @misc{singh2021_data,
        title={Interval Timing Task},
        author={Singh, Arun and Cole, Rachel and Espinoza, Arturo and Wessel, Jan R. and Cavanagh, Jim and Narayanan, Nandakumar},
        publisher={OpenNeuro},
        doi={10.18112/openneuro.ds004579.v1.0.0},
        url={https://openneuro.org/datasets/ds004579/versions/1.0.0}
    }
    """
    licence: tp.ClassVar[str] = "CC0-1.0"
    description: tp.ClassVar[str] = (
        "EEG recordings in Parkinson's disease patients and healthy controls during "
        "an interval timing task (OpenNeuro ds004579)."
    )
    _TASK: tp.ClassVar[str] = "IntervalTiming"
    # BIDS ``_events.tsv`` "value" markers -> integer codes.
    _CODE_MAPPING: tp.ClassVar[dict[str, int]] = {
        "S  1": 1,
        "S  2": 2,
        "S  3": 3,
        "S  4": 4,
        "S  5": 5,
        "S  6": 6,
        "S  7": 7,
        "S255": 8,
        "boundary": 9,
        "R  3": 10,
    }
    # BIDS ``_events.tsv`` "value" markers -> human-readable descriptions.
    _DESCRIPTION_MAPPING: tp.ClassVar[dict[str, str]] = {
        "S  1": "short_interval_instruction",  # 1 s
        "S  2": "long_inverval_instruction",  # 1 s
        "S  3": "interval_start",  # Start of interval (blue rectangle shown)
        # Lasts 8-10 s for short intervals and 18-20 s for long intervals
        "S  4": "spacebar_press",  # XXX Replace with Button event of right duration
        "S  5": "spacebar_release",
        "S  6": "distracting_vowel",
        # XXX Interval end is not available in the dataset
        "S  7": "trial_feedback",  # On 15% of the trials
        "S255": "end_of_last_trial",
        # Not described in README.md
        "boundary": "unknown",  # BrainVision "New Segment/"
        "R  3": "unknown",  # BrainVision "Response/R  3"
    }
    # TODO: populate after downloading the OpenNeuro data, e.g.
    #   python -c "from neuralfetch.utils import update_source_info; \
    #       update_source_info('Singh2021Timing_Openneuro')"
    _info: tp.ClassVar[study.StudyInfo | None] = None

    def _download(self, overwrite: bool = False) -> None:
        download.Openneuro(study="ds004579", dset_dir=self.path).download(
            overwrite=overwrite
        )

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        """Returns a generator of all recordings"""
        participants = pd.read_csv(
            self.path / "download" / "participants.tsv", sep="\t"
        )
        diagnosis_map = {"PD": "parkinsons", "Control": "control"}
        for row in participants.itertuples():
            sub_id = str(row.participant_id).removeprefix("sub-")
            fpath = (
                self.path
                / "download"
                / f"sub-{sub_id}"
                / "eeg"
                / f"sub-{sub_id}_task-{self._TASK}_eeg.set"
            )
            if not fpath.exists():
                continue
            yield dict(
                subject=sub_id,
                task=self._TASK,
                diagnosis=diagnosis_map.get(row.GROUP, row.GROUP),
            )

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        bids_path = BIDSPath(
            subject=timeline["subject"],
            task=timeline["task"],
            root=self.path / "download",
            datatype="eeg",
            suffix="eeg",
            extension=".set",
        )
        events_path = bids_path.copy().update(suffix="events", extension=".tsv")

        events = pd.read_csv(events_path.fpath, sep="\t")
        events.rename(columns={"onset": "start"}, inplace=True)
        events["type"] = "Stimulus"
        events["code"] = events["value"].map(self._CODE_MAPPING)
        events["description"] = events["value"].map(self._DESCRIPTION_MAPPING)
        events["duration"] = pd.to_numeric(events["duration"], errors="coerce")
        events = events[["type", "start", "duration", "code", "description"]]

        eeg = dict(type="Eeg", filepath=str(bids_path.fpath), start=0)
        events = pd.concat([pd.DataFrame([eeg]), events], ignore_index=True)
        return events
