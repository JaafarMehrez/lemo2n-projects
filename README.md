# lemo2n-projects

**A concise guide for contributors** — how to add your project folder to this repository by forking, committing, and opening a pull request (PR).

---

## Quick overview

1. Fork the repository on GitHub (top-right **Fork**).
2. Clone *your fork* to your machine.
3. Create a feature branch for your work.
4. Add a project folder and include a short `README.md` inside it.
5. Commit with a descriptive message.
6. Push the branch to your fork.
7. Open a PR against `lemo2n-Lab/lemo2n-projects` and request review.

---

## 1. Fork the repository
On GitHub, open `https://github.com/lemo2n-Lab/lemo2n-projects` and click **Fork** (top-right). This creates a copy under your account where you can push changes.

---

## 2. Clone your fork
Clone your fork to your local machine and enter the repo directory:

```bash
# replace <your-username> with your GitHub username
git clone git@github.com:<your-username>/lemo2n-projects.git
cd lemo2n-projects
```

> If you accidentally cloned the upstream repo, change remotes:

```bash
# rename the upstream origin, add your fork as origin
git remote rename origin upstream
git remote add origin git@github.com:<your-username>/lemo2n-projects.git
git push -u origin main
```

---

## 3. Create a feature branch
Create a dedicated branch for your work (do not work directly on `main`):

```bash
git checkout -b feature/<project-name>-<initials>
```

**Examples:**
- `feature/adsorption-Pt`
- `feature/si-interface`

---

## 4. Add your project folder and files
Create a folder named after your project and add all relevant files (input file, geometry, main output file). Include a short `README.md` inside that folder describing the contents and how to reproduce results.

```bash
mkdir MyProjectName
# Add files to the folder (examples)
cp /path/to/results MyProjectName/
echo "# MyProjectName\n\nShort description and reproduction steps." > MyProjectName/README.md
```

**Guidelines**
- Keep each project self-contained.
- Avoid committing sensitive data (credentials, private keys).
- Avoid very large binary files (>100 MB). Use institutional storage or Git LFS if necessary and agreed by the group.

---

## 5. Stage and commit your changes
Stage only the files you intend to include, then commit with a clear, conventional message:

```bash
git add MyProjectName
git commit -m "feat(MyProjectName): add initial project files and README"
```

**Commit message style**
- Use imperative mood (`add`, `fix`, `update`).
- Optionally prefix with `feat()`, `fix()`, `doc()`, etc., and include the project name.

---

## 6. Push your branch to GitHub
Push your branch to your fork:

```bash
git push -u origin feature/<project-name>-<initials>
```

---

## 7. Open a Pull Request (PR)
Open a PR from your branch in your fork to the upstream repository (`lemo2n-Lab/lemo2n-projects`), targeting the `main` branch (or whatever branch maintainers specify).

**On GitHub (web)**
- Go to your fork, switch to your branch, click **Compare & pull request**.
- Write a clear PR title and description (see PR template below).
- Request reviewers (e.g., `@JaafarMehrez`, `@Wu-maokun`) and add relevant labels if available.
- Submit the PR.

**Using GitHub CLI (optional)**

```bash
gh pr create --base lemo2n-Lab:main --head <your-username>:feature/<project-name>-<initials> \
  --title "feat(MyProjectName): short summary" \
  --body "Longer description and reproduction steps, links, and notes."
```

---

## PR template — suggested content
Include the following in your PR body to help reviewers:

- **What**: brief summary of what you added (folder, core files).
- **Why**: why it belongs in this repository.
- **How to reproduce**: quick steps or reference to the project `README.md`.
- **Notes**: large files, external datasets, or use of Git LFS.

**Checklist**
- [ ] Project folder includes `README.md` with reproduction steps
- [ ] No sensitive data or private keys committed
- [ ] File sizes are acceptable (or large files referenced externally)
- [ ] Any scripts include usage examples

---

## Best practices & conventions
- **Branch names:** `feature/`, `fix/`, `doc/` + descriptive name and initials (e.g., `feature/polaron-model-jd`).
- **Commits:** keep them focused and small. Use meaningful messages.
- **Project README:** short description, key files, reproduction steps, dependencies.
- **Large files:** prefer external storage (institutional server, Zenodo, Figshare) and link from README. Use Git LFS only if agreed by the group.
- **Licensing:** if your project code requires a license file, include it or note restrictions in the project README.

---

## Troubleshooting
- **`git push` denied:** Ensure you pushed to *your fork* (`origin`) and set the upstream branch with `-u` the first time.
- **Accidentally committed sensitive data:** Contact maintainers immediately. Removing sensitive data from history requires tools like `git filter-repo` or `BFG Repo-Cleaner` (these rewrite history — coordinate with maintainers).
- **Large file error (GitHub >100 MB):** Move the file out of the repo and link to an external store, or use Git LFS if approved.

---

## Support / contacts
If you need help, mention the maintainers in your PR or contact them directly:

- Jaafar — `@JaafarMehrez`
- Wu Maokun - `@Wu-maokun`

---

Thank you for contributing. Clear, well-documented project folders make the repository useful and easy to navigate.

---

