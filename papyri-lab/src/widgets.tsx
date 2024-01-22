import { requestAPI } from './handler';
import { MyPapyri, RENDERERS, SearchContext } from './papyri-comp';
import { ReactWidget } from '@jupyterlab/apputils';
import { ThemeProvider } from '@myst-theme/providers';
import React, { useState } from 'react';

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
const PapyriComponent = (): JSX.Element => {
  // list of a few pages in the database that matches
  // the current query
  const [possibilities, setPossibilities] = useState([]);
  const [navs, setNavs] = useState<string[]>([]);
  const [root, setRoot] = useState({});

  const [searchTerm, setSearchTerm] = useState('');

  // callback when typing in the input field.
  const onChange = async (event: any) => {
    setSearchTerm(event.target.value);
    search(event.target.value);
  };

  const back = async () => {
    navs.pop();
    const pen = navs.pop();
    if (pen !== undefined) {
      console.log('Setting search term', pen);
      await setSearchTerm(pen);
      console.log('... and searchg for ', pen);
      await search(pen);
    }
  };

  const search = async (query: string) => {
    const res = await requestAPI<any>('get-example', {
      body: query,
      method: 'post'
    });
    console.log('Got a response for', query, 'res.body=', res.body);
    // response has body (MyST–json if the query has an exact match)
    // and data if we have some close matches.
    if (res.body !== null) {
      setNavs([...navs, query]);
      setRoot(res.body);
      setPossibilities([]);
    } else {
      setRoot({});
      setPossibilities(res.data);
    }
  };

  const onClick = async (query: string) => {
    console.log('On click trigger', query);
    setSearchTerm(query);
    try {
      search(query);
    } catch (e) {
      console.error(e);
    }
    return false;
  };

  return (
    <React.StrictMode>
      <input onChange={onChange} value={searchTerm} />
      <button onClick={back}>Back</button>
      <ul>
        {possibilities.map(e => {
          return (
            <li>
              <a
                href={e}
                onClick={async () => {
                  await onClick(e);
                }}
              >
                {e}
              </a>
            </li>
          );
        })}
      </ul>
      <div className="view">
        <SearchContext.Provider value={onClick}>
          <ThemeProvider renderers={RENDERERS}>
            <MyPapyri node={root} />
          </ThemeProvider>
        </SearchContext.Provider>
      </div>
    </React.StrictMode>
  );
};

// This seem to be the way to have an adapter between lumino and react, and
// allow to render react inside a JupyterLab panel
export class PapyriPanel extends ReactWidget {
  constructor() {
    super();
    this.addClass('jp-ReactWidget');
    this.id = 'papyri-browser';
    this.title.label = 'Papyri browser';
    this.title.closable = true;
  }

  render(): JSX.Element {
    return <PapyriComponent />;
  }
}
