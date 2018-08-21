This file describes the main differences between the Dec. 2002 release of
PropBank and the current, Feb. 2004, release.

[Note from CoNLL] : Numbers refer to the whole PropBank release, not to the
 selected sections for the CoNLL-2004 Shared Task. 

1) Coverage
------------

Dec. 2002 - 72,109  total propositions
Feb. 2004 - 104,952 total propositions

The annotations now span the entire WSJ section of the Penn TreeBank II corpus.
There are, however, still some minor gaps in the data.  We are still working
on the analysis of some of the trickier verbs (e.g. "keep").  In addition,
there are some verbs which have not been fully adjudicated (make, go, take).
As this release ONLY contains adjudicated annotations (to ensure high quality
and consistency), we expect the final release to gain about 5,000 annotations.


2) Adjudication
----------------

Dec. 2002 - 11,092  adjudicated propositions
Feb. 2004 - 104,952 adjudicated propositions

All of the annotations in the current release have gone through a rigorous
process of double-annotation/adjudication.  As such we expect the annotations
to be more consistent and coherent than those contained in the Dec. 2002
release.


3) Roleset Tagging
-------------------

Dec. 2002 - 12,468 polysemous instances tagged with unique roleset.
Feb. 2004 - 44,021 polysemous instances tagged with unique roleset.

Besides far greater coverage, the roleset tags in the current release are
all the result of a double-blind/adjudicated annotation process (whereas
the Dec. 2002 roleset tags were only single annotated).  There are still
3,561 instances which need roleset tags - they have been double annotated 
but not adjudicated.  The agreement rate on roleset tagging is > 91%.
   

4) Quality Control
-------------------

Besides only including gold standard annotations, we have taken other 
steps to ensure that the data in this release is of the highest possible
quality.  We have run error searching programs to ensure a) that the roleset
tagging is consistent with the argument labels assigned, b) that all PP
arguments are annotated on the NP and that the preposition is included
in the label, c) that all sentential arguments are annotated at the level
of the SBAR node.  We have also done additional hand-checks of those types
of instances that are particularly error-prone, throwing out all instances
that were not fully consistent.  Also, we have removed the inflection
tagging from this release as it had not been done consistently in the past.  
We are currently doing inflection tagging as a separate task at an 
inter-annotator agreement rate of > 93% and at very high speeds.



6) Changes in Frames Files
-------------------------

Since the Dec. 2002 release, the frames files have undergone many
changes.  For example, the aspectual usages (begin, start, end etc)
have been reanalyzed, reframed, and reannotated.  In addition, the 
numbering of the arguments has been more standardized (compare call.xml
in the old and new releases), and rolesets have been added.  For these
reasons the new data completely supercedes the data from the previous 
release.


7) Changes in Annotation Guidelines
------------------------------------

a) PRD tags are no longer present on numbered arguments (their application
   had been inconsistent)
b) PRP tags are no longer used (instead, we used PNC for Purpose and CAU 
   for Cause)
c) ARGM's must always have secondary labels (e.g. TMP, LOC, MNR, etc).
d) Sentential arguments are now tagged at the SBAR level (to include the 
   complementizer).


===========================================================================
                  
