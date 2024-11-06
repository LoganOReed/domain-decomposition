# DD Lecture (80 minutes)
## Alternating Schwarz Method (20-30 min)
1. Explain algorithm and do very simple example (5-10 min)
2. Give geometric sketch for convergence in 1d and write convergence factor (5 min)
3. Explain underlying assumptions and outline how they relate to multiscale (5 min)
4. Give general guidelines for the various families of multiscale (5-10 min)
5. Explain cons and motivate next method (2 min)

## P.L. Lions' (19-30 min)
1. Explain algorithm and Robin BCs, and note it works with overlapping and nonoverlapping (5-10 min)
2. Convergence Proof (10-20 min)
3. Give a couple of examples that Lions' handles better (2-5)
3a. boundary/interior layer problems
4. Introduce difficulty with implementing transmission conditions (1-2 min)

## Discretization and Implementation (35-50 min)
1. Explain difficulty with outer normal derivative (2-5 min)
2. Go over trick for nonoverlapping Lions' (for HW) (10-15 min)
3. Explain why above won't work for overlapping, and go through various discretization options (5-10 min)
3a. uniform
3b. uniform variable resolution
3c. arbitrary mesh
4. Emphasize the benifits/tradeoffs balancing when choosing. (2-5 min)
5. Mention that all of the DD methods can be understood as preconditioners for Krylov methods (5-10)
    - This will make code significantly faster, recommended if having issues
6. Outline Equivalence between Optimized Restricted Additive Schwarz (ORAS) and (overlapping) P.L. Lions' (10-20 min)


## Show Code Examples (5-10 min)
TODO: Once HW is decided, add the rest here

## ONLY IF ADDITIONAL TIME 
1. Mention a couple of specific use cases found for multiscale Lions' with Robin BCs
