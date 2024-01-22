import { requestAPI } from './handler';
import { MyPapyri, RENDERERS, SearchContext } from './papyri-comp';
import { ReactWidget } from '@jupyterlab/apputils';
import { ThemeProvider } from '@myst-theme/providers';
import React from 'react';

// this is a papyri react component that in the end should
// have navigation UI and a myst renderer to display the
// documentation.
//
// It is pretty bare bone for now, but might have a toolbar later.
//
// It would be nice to have this outside of the JupyterLab extension to be reusable
//
// I'm going to guess it will need some configuration hook for when we click on links.
//
//
class PapyriComponent extends React.Component {
  state = {
    possibilities: [],
    navs: [],
    root: {},
    searchterm: ''
  };
  constructor(props: any) {
    super(props);
    this.state = {
      possibilities: [],
      navs: [],
      root: {},
      searchterm: ''
    };
  }

  setPossibilities(pos: any) {
    this.setState({ possibilities: pos });
  }

  setNavs(navs: any) {
    this.setState({ navs: navs });
  }

  setRoot(root: any) {
    this.setState({ root: root });
  }

  setSearchTerm(searchterm: string) {
    this.setState({ searchterm: searchterm });
  }

  async handleInputChange(event: any) {
    console.log('on change, this is', this);
    this.setSearchTerm(event.target.value);
    this.search(event.target.value);
  }

  async back() {
    this.state.navs.pop();
    const pen = this.state.navs.pop();
    if (pen !== undefined) {
      console.log('Setting search term', pen);
      await this.setSearchTerm(pen);
      console.log('... and searchg for ', pen);
      await this.search(pen);
    }
  }
  async search(query: string) {
    const res = await requestAPI<any>('get-example', {
      body: query,
      method: 'post'
    });
    console.log('Got a response for', query, 'res.body=', res.body);
    // response has body (MyST–json if the query has an exact match)
    // and data if we have some close matches.
    if (res.body !== null) {
      this.setNavs([...this.state.navs, query]);
      this.setRoot(res.body);
      this.setPossibilities([]);
    } else {
      this.setRoot({});
      this.setPossibilities(res.data);
    }
  }

  async onClick(query: string) {
    console.log('On click trigger', query, 'this is', this);

    this.setSearchTerm(query);
    try {
      this.search(query);
    } catch (e) {
      console.error(e);
    }
    return false;
  }

  render(): JSX.Element {
    return (
      <React.StrictMode>
        <input
          onChange={this.handleInputChange.bind(this)}
          value={this.state.searchterm}
        />
        <button onClick={this.back}>Back</button>
        <ul>
          {this.state.possibilities.map((e: any) => {
            return (
              <li>
                <a
                  href={e}
                  onClick={async () => {
                    await this.onClick(e);
                  }}
                >
                  {e}
                </a>
              </li>
            );
          })}
        </ul>
        <div className="view">
          <SearchContext.Provider value={this.onClick.bind(this)}>
            <ThemeProvider renderers={RENDERERS}>
              <MyPapyri node={this.state.root} />
            </ThemeProvider>
          </SearchContext.Provider>
        </div>
      </React.StrictMode>
    );
  }
}

// This seem to be the way to have an adapter between lumino and react, and
// allow to render react inside a JupyterLab panel
export class PapyriPanel extends ReactWidget {
  comp: any;
  constructor() {
    super();
    this.addClass('jp-ReactWidget');
    this.id = 'papyri-browser';
    this.title.label = 'Papyri browser';
    this.title.closable = true;
    this.comp = React.createRef();
  }

  updateSeachTerm(str: string) {
    this.comp.current.setSearchTerm(str);
    this.comp.current.search(str);
  }

  render(): JSX.Element {
    return <PapyriComponent ref={this.comp} />;
  }
}
