# StructuredFFN

![StructuredFFN](https://user-images.githubusercontent.com/13270895/89133355-26b3af80-d4e9-11ea-81cd-eacaa9c78320.png)

StructuredFFN is a permanent dark theme of the Poole theme by [@mdo](https://github.com/mdo). I made the theme darker, inspired by [Derek Kedziora's site](https://derekkedziora.com/). Unlike default Poole that utilizes CSS media queries to activate dark mode, the theme will stay dark regardless of the user's preference.

- I added a navbar that is easily customizable. Check out [Development](#development) to see how.
- I also got rid of the "tagline" in the navbar. I think it looks cleaner without it.
- Finally, I changed the default font size to 20px. I have 20/20 vision and still thought the original font size was too small.

That's it! I tried to be least intrusive as possible to the Poole code base.

**I noticed that Poole's documentation is slightly outdated and misleading. This documentation will try to address most, if not all, of these issues.**

---

## Contents

- [Usage](#usage)
- [Development](#development)
- [Author](#author)
- [License](#license)

## Usage

### 1. Install dependencies

Poole is built on Jekyll and uses its built-in SCSS compiler to generate our CSS. Before getting started, you'll need to install the Jekyll gem and related dependencies:

```bash
$ gem install jekyll jekyll-gist jekyll-sitemap jekyll-seo-tag
```

### 2. Install bundler

You must have bundler installed. If you already have bundler installed, please skip this step.

```bash
# Update Rubygems
$ gem update --system
# Update bundler
$ gem install bundler
```

### 3. Running locally

To see your Jekyll site with Poole applied, start a Jekyll server. In Terminal, from `/dark-poole` (or whatever your Jekyll site's root directory is named):

```bash
$ bundle exec jekyll serve
```

Open <http://localhost:4000> in your browser, and voilà.

### 4. Serving it up

If you host your code on GitHub, you can use [GitHub Pages](https://pages.github.com) to host your project.

1. Fork this repo and switch to the `gh-pages` branch.
1. If you're [using a custom domain name](https://help.github.com/articles/setting-up-a-custom-domain-with-github-pages), modify the `CNAME` file to point to your new domain.
1. If you're not using a custom domain name, **modify the `url` in `_config.yml`** to point to your GitHub Pages URL. Example: for a site hosted at `username.github.io`, use `http://username.github.io`.
1. If you want to use your repo name as a base url, **set the `url`** to your repo link and **set the `baseurl`** to your repo name in **`_config.yml`**. Example: for site hosted on `https://username.github.io/dark-poole`, set `url` as `https://username.github.io/dark-poole` and `baseurl` as `/dark-poole`.
1. Done! Head to your GitHub Pages URL or custom domain.

No matter your production or hosting setup, be sure to verify the `baseurl` option file and `CNAME` settings. Not applying this correctly can mean broken styles on your site.

### 5. Pagination for sites with base urls

If you are using a base url for your site, (for example, hosted on `https://username.github.io/dark-poole`) you have to make some changes to get jekyll-pagination to work correctly:

In `_config.yml`, add this line:

```yaml
paginate_path: "/baseurl/page:num/"
```

In `archive.md`, add `{{ site.baseurl }}` before `{{ post.url }}`

```html
<!-- Add "{{ site.baseurl }}" -->
<li><a href="{{ site.baseurl }}{{ post.url }}">{{ post.title }}</a></li>
```

In `index.html`, remove the `prepend:`:

```html
<!-- Remove "prepend:" in "prepend: relative_url" -->
<a
  class="pagination-item newer"
  href="{{ paginator.previous_page_path | relative_url }}"
  >Newer</a
>
```

## Development

Poole has two branches, but only one is used for active development.

- `master` for development. **All pull requests should be to submitted against `master`.**
- `gh-pages` for hosted demo **Please avoid using this branch.**

CSS is handled via Jeykll's built-in Sass compiler. Source Sass files are located in `_sass/`, included into `styles.scss`, and compile to `styles.css`.

### Customize Navbar

You can easily customize the navbar by tweaking the `_config.yml` file. Simply change the title and url of each of the nav elements, or add more. The order will be preserved in the site.

```yaml
nav:
  - title: Blog
    url: /archive

  - title: About
    url: /about
```

## Author

**Mark Otto**

- <https://github.com/mdo>
- <https://twitter.com/mdo>

## License

Open sourced under the [MIT license](LICENSE.md).

<3
