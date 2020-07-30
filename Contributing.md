# Contributing

This file is not rendered correctly in Gogs.
[View on github](https://github.com/pepper-jk/Hierarchical-Federated-Learning-Quantization/blob/TK_master/Contributing.md).

## How to contribute

### You have access to the repo?
Great, just create your own working branches and follow the workflow done below.

### You do not have access to the repo?
No problem, just create a fork and send in pull requests.
Follow the workflow below, just think "pull request", when it say `merge`.

## Workflow

- never force push (`push -f`) into master
- only force push (`push -f`) into master under the following circumstances
  - to bringing in changes from [upstream](https://github.com/wesleyjtann/Hierarchical-Federated-Learning-Quantization)
  - to provide a curcial bugfix for a commit (via `commit --ammend` or `rebase -i`)
  - always inform all direct contributors beforehand, so they can rebase their branches afterwards
- `rebase` working branches on master on a **regular bases**
    ```shell
    # check if origin/master has changes:
    $ git fetch
    $ git checkout master
    $ git pull
    # rebase current branch and update its remote counterpart
    $ git checkout <working_branch>
    $ git rebase master
    $ git push -f
    ```
- always `rebase` on the most recent master **before merging into master**
- **no broken commits** in master
  - **Every commit** that is merged into master should work on its own, meaning no functionality should be sacrificed in one and fixed in the commit after.
  - in these cases always `squash` commits before merging.
  - this can be done with `git rebase -i master`
    ```
    pick   <hash> feature_1
    squash <hash> feature_1_bugfix_1
    squash <hash> feature_1_bugfix_2
    pick   <hash> feature_2
    squash <hash> feature_2_bugfix
    ```
- a working branch should be prepared with `rebase -i` before a `merge`
- each commit name should explain what was implemented
  - referencing function and file names that have speaking names helps

## Make your live easier

- try to keep each commit **minimal**
  - this will make `rebase` and `merge` easier
  - commit as less code as possible at once
    - minimal number of files
    - minimal lines of code that makes sense
- keep all working branches **as consistent as possible**
  - for **less conflicts** during `rebase` and `merge`
  - some examples, where `a`, `e`, `g`, and `i` are the first commits of a new working branch
  - consistent:
    ```
    master
          \
           a - b - c - d
                        \
                         e - f
                              \
                               g - h
                                    \
                                     i - j - k
    ```
  - less consistent:
    ```
    master
          \
           a - b - c - d
            \           \
             e - f       g - h
                  \
                   i - j - k
    ```
  - not consistent:
    ```
    master
          \
           a - b - c - d
            \
             e - f
             | \
              \ g - h
               \
                i - j - k
    ```
- if a underlying working branch changes like so
    ```
    master
        \
         a - b - c - d                # old working branch 1
             |        \
             |         e - f          # old working branch 2
             \
              g - c' - d'             # new working branch 1
    ```
  - use [rebase --onto](https://dev.to/martinbelev/how-to-effectively-use-git-rebase-onto-5b85) to get
    ```
    master
        \
         a - b - c - d                # old working branch 1
             |        \
             |         e - f          # old working branch 2
             \
              g - c' - d'             # new working branch 1
                        \
                         e' - f'      # new working branch 2
    ```
- also try to order the commits in a pattern that makes sense
  - for example group them by feature
    ```
    pick <hash> add feature_1 to lib
    pick <hash> use feature_1 in main
    pick <hash> add feature_2 to other_lib
    pick <hash> use feature_2 in main
    ```
    or by categories such as "adding lib function" and "using libfunction" and create "waves"
    ```
    pick <hash> add feature_1 to lib
    pick <hash> add feature_2 to other_lib
    pick <hash> use feature_1 in main
    pick <hash> use feature_2 in main
    ```
  - **NOT**
    ```
    pick <hash> add feature_1 to lib
    pick <hash> add feature_2 to other_lib
    pick <hash> other commit
    pick <hash> use feature_1 in main
    pick <hash> use feature_2 in main
    ```
    and
    ```
    pick <hash> add feature_1 to lib
    pick <hash> other commit 1
    pick <hash> use feature_1 in main
    pick <hash> other commit 2
    pick <hash> add feature_2 to other_lib
    pick <hash> other commit 3
    pick <hash> other commit 4
    pick <hash> use feature_2 in main
    ```
