## What this changes

<!-- One or two sentences. What does this add, fix or change for users? -->

## Checklist

- [ ] `pytest` passes locally (the fast suite — no flags needed)
- [ ] New behaviour has a test, or this is a change no test could catch (say which)
- [ ] **Adding a dataset?** `pytest tests/test_dataset_contracts.py` passes.
      It checks the metadata contracts `evaluate()` relies on and catches
      most new-wrapper mistakes before review.
- [ ] **Golden-value test failed?** The numbers moved for a reason you
      understand, and you have explained it below. Regenerating
      `tests/data/golden/` to turn a test green without that explanation
      discards the only thing those fixtures watch.
- [ ] **Removed an `xfail`?** A strict xfail turning red means you fixed the
      bug it documented. Delete the marker and note the fix below.

## Numerical changes

<!-- Delete if nothing numerical moved. Otherwise: which golden values
     changed, by how much, and why that is correct. -->

## Notes for the reviewer

<!-- Anything you want a second opinion on, or decisions worth questioning. -->
