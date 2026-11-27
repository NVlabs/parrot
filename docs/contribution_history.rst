Recovered contribution history
==============================

This record preserves seven merged contributions by ``0z5a`` whose original
GitHub pull-request pages were unavailable when checked on 26 November 2026.
The cause of the visibility change has not been established. All seven merge
commits remain in ``main`` at the verified revision
``b3690d287d264ce9e788be4a9945d0a221e74ce0``.

A saved GitHub API response from 20 November 2026 confirms the original PR
numbers, titles, contributor, and merge timestamps. Tracking issue
`#136 <https://github.com/NVlabs/parrot/issues/136>`_ records the preservation
work. This page can remain useful if the original PR pages are restored.

Evidence and provenance
-----------------------

* :download:`Original metadata recorded on 20 November 2026 <_static/pr-history/github-metadata-2026-11-20.json>`
* :download:`Contribution records and file checksums <_static/pr-history/records.json>`

Original description snapshots were recovered for PRs #108, #124, and #133.
For PRs #125, #126, #131, and #132, the original Git commit messages are
preserved as the source for the summaries below; their original PR descriptions
have not been recovered. Snapshots may predate later edits. Original PR comments
and reviews have not been recovered.

Each merged diff compares the merge commit with its first parent. It records
the change integrated into the repository, including any merge resolution.
The original contributor and existing Git commit identities are preserved.
Validation results in recovered descriptions describe historical runs reported
at the time and have not been rerun for this documentation record.

.. _pr-history-108:

PR #108: Propagate explicit CUDA stream through segmented reductions
--------------------------------------------------------------------

Forward an explicit CUDA stream through segmented-reduction scratch-size queries, launches, and row/column callers. The recovered description documents the synchronous allocation policy and validation limitations at the time.

* Original PR: `#108 <https://github.com/NVlabs/parrot/pull/108>`_
* Original contributor: ``0z5a``
* Merged at: ``2026-09-28T14:16:29Z``
* Merge commit: `b144a97 <https://github.com/NVlabs/parrot/commit/b144a975a5a34c8d8dca41d15790f0eb3561cd22>`_
* Merged head: ``8cb2cef457acf661d48caffbc2599d82cbb6a695``
* :download:`Exact merged diff <_static/pr-history/pr-108-merged.diff>`
* :download:`Original description snapshot <_static/pr-history/pr-108-original-description.txt>`, captured ``2026-09-28T14:03:09.574Z``

.. _pr-history-124:

PR #124: Apply keep/filter masks in map, map_adj, rand and array map2
-----------------------------------------------------------------------

Materialize keep/filter masks before map, adjacent map, random generation, and array map2 so these operations consume the logical selected elements. For array map2, apply a mask on either operand.

* Original PR: `#124 <https://github.com/NVlabs/parrot/pull/124>`_
* Original contributor: ``0z5a``
* Merged at: ``2026-11-02T21:46:24Z``
* Merge commit: `3ca28b4 <https://github.com/NVlabs/parrot/commit/3ca28b4b99e046a049e69a93e09b9cfc8fbd8715>`_
* Merged head: ``4cb7d477dd9d475aea4d3deca52abddabe9eedb1``
* :download:`Exact merged diff <_static/pr-history/pr-124-merged.diff>`
* :download:`Original description snapshot <_static/pr-history/pr-124-original-description.txt>`, captured ``2026-11-02T20:26:23.287Z``

.. _pr-history-125:

PR #125: Return min and max by value so lazy chains do not dangle
-------------------------------------------------------------------

Return min and max results by value so lazy expressions do not retain dangling references. Correct the affected chained-operation expectation and check that the functors return values.

* Original PR: `#125 <https://github.com/NVlabs/parrot/pull/125>`_
* Original contributor: ``0z5a``
* Merged at: ``2026-11-03T18:17:41Z``
* Merge commit: `c5acfcb <https://github.com/NVlabs/parrot/commit/c5acfcb882dd880317af7cf3c2a6e758df10fba9>`_
* Merged head: ``d5bafbb991203a7f23b3431191372702b5afaa0a``
* :download:`Exact merged diff <_static/pr-history/pr-125-merged.diff>`
* :download:`Original commit message <_static/pr-history/pr-125-commit-message.txt>`; original PR description unavailable

.. _pr-history-126:

PR #126: Apply keep/filter masks in structural operations and keep operands alive
-----------------------------------------------------------------------------------

Apply pending masks before structural operations and retain storage for both operands in lazy two-array operations. The original commit explains masked-size mismatches, out-of-bounds accesses, and temporary-operand lifetimes.

* Original PR: `#126 <https://github.com/NVlabs/parrot/pull/126>`_
* Original contributor: ``0z5a``
* Merged at: ``2026-11-03T18:16:43Z``
* Merge commit: `f2d4792 <https://github.com/NVlabs/parrot/commit/f2d47920269dd86f72a38f4f82281fd02c465525>`_
* Merged head: ``1be3bc6477b00b019536f3225f176a154751e9f7``
* :download:`Exact merged diff <_static/pr-history/pr-126-merged.diff>`
* :download:`Original commit message <_static/pr-history/pr-126-commit-message.txt>`; original PR description unavailable

.. _pr-history-131:

PR #131: Treat the keep mask as a predicate in masked sum
-----------------------------------------------------------

Treat every nonzero mask value as keep during masked sum. Select the value or zero instead of multiplying by the mask, avoiding weighted results and contamination from excluded infinities or NaNs.

* Original PR: `#131 <https://github.com/NVlabs/parrot/pull/131>`_
* Original contributor: ``0z5a``
* Merged at: ``2026-11-08T16:17:12Z``
* Merge commit: `8cc2f28 <https://github.com/NVlabs/parrot/commit/8cc2f286c31c7179fe8fbd8f2da486ea5342cc0d>`_
* Merged head: ``3a98e54432f4031ede165236531b5a2238c40f89``
* :download:`Exact merged diff <_static/pr-history/pr-131-merged.diff>`
* :download:`Original commit message <_static/pr-history/pr-131-commit-message.txt>`; original PR description unavailable

.. _pr-history-132:

PR #132: Apply the mask and return a copy from value()
--------------------------------------------------------

Implement value() through front() so it honors masks and returns a host value instead of a device reference into temporary storage. The original commit describes empty-mask behavior and result-type compatibility.

* Original PR: `#132 <https://github.com/NVlabs/parrot/pull/132>`_
* Original contributor: ``0z5a``
* Merged at: ``2026-11-08T16:18:11Z``
* Merge commit: `ea6d469 <https://github.com/NVlabs/parrot/commit/ea6d469a1b4d65ca480628e7e82e459ea2f25461>`_
* Merged head: ``b21b2820e6450504faef6b554285da517d437c94``
* :download:`Exact merged diff <_static/pr-history/pr-132-merged.diff>`
* :download:`Original commit message <_static/pr-history/pr-132-commit-message.txt>`; original PR description unavailable

.. _pr-history-133:

PR #133: Broadcast a rank-0 operand to an empty array without reading it
--------------------------------------------------------------------------

Use the non-scalar operand size when broadcasting a rank-zero result against an array. This preserves an empty result for an empty array and avoids reading beyond its storage.

* Original PR: `#133 <https://github.com/NVlabs/parrot/pull/133>`_
* Original contributor: ``0z5a``
* Merged at: ``2026-11-10T00:09:06Z``
* Merge commit: `8740d1e <https://github.com/NVlabs/parrot/commit/8740d1e380486c66410e6e91e7cf349c734f20c2>`_
* Merged head: ``2f2688dd4d37532d59fe75062db5a2835511671d``
* :download:`Exact merged diff <_static/pr-history/pr-133-merged.diff>`
* :download:`Original description snapshot <_static/pr-history/pr-133-original-description.txt>`, captured ``2026-11-10T00:03:52.228Z``
