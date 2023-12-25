// Global and other papyri-myst related componets
import { DEFAULT_RENDERERS, MyST } from 'myst-to-react';
import React from 'react';
import { createContext, useContext } from 'react';

export const SearchContext = createContext(async (query: string) => {
  return true;
});

const MyLink = ({ node }: any) => {
  const onSearch = useContext(SearchContext);
  const parts = node.url.split('/');
  const search_term = parts[parts.length - 1];
  const f = (q: string) => {
    console.log('sustom onclick', q, onSearch);
    onSearch(q);
  };

  return (
    <a onClick={() => f(search_term)}>
      <MyST ast={node.children} />
    </a>
  );
};

const DefaultComponent = ({ node }: { node: any }) => {
  if (!node.children) {
    return <span>{node.value}</span>;
  }
  return (
    <div className="not-implemented">
      <MyST ast={node.children} />
    </div>
  );
};

const MUnimpl = ({ node }: { node: any }) => {
  if (!node.children) {
    return <span>{node.value}</span>;
  }
  return (
    <div className="not-implemented unimpl">
      <MyST ast={node.children} />
    </div>
  );
};

const Param = ({ node }: { node: any }) => {
  return (
    <>
      <dt>
        {node.param}: {node.type_}
      </dt>
      <dd>
        {node.desc.map((sub: any) => (
          <MyST ast={sub} />
        ))}
      </dd>
    </>
  );
};
const Parameters = ({ node }: { node: any }) => {
  return (
    <dl>
      {node.children.map((item: any) => (
        <MyST ast={item} />
      ))}
    </dl>
  );
};

const DefList = ({ node }: { node: any }) => {
  return (
    <dl>
      {node.children.map((item: any) => (
        <>
          <dt>
            <MyST ast={item.dt} />
          </dt>
          <dd>
            {item.dd.map((sub: any) => (
              <MyST ast={sub} />
            ))}
          </dd>
        </>
      ))}
    </dl>
  );
};

// render a single parameter in a signature
const ParameterNodeRenderer = ({ node }: { node: any }) => {
  if (node.kind === '/') {
    return <span>/</span>;
  }
  let comp = [];
  let name = '';
  if (node.kind === 'VAR_POSITIONAL') {
    name = '*';
  }
  if (node.kind === 'VAR_KEYWORD') {
    name += '**';
  }
  name += node.name;

  comp = [<span>{name}</span>];

  if (node.annotation.type !== 'Empty') {
    comp.push(<span className="type-ann">{': ' + node.annotation.data}</span>);
  }
  if (node.default.type !== 'Empty') {
    comp.push(<span className="default-value">{'=' + node.default.data}</span>);
  }

  return <>{comp}</>;
};

const SignatureRenderer = ({ node }: { node: any }) => {
  let prev = '';
  const acc = [];
  for (let i = 0; i < node.parameters.length; i++) {
    const p = node.parameters[i];
    if (p.kind !== 'POSITIONAL_ONLY' && prev === 'POSITIONAL_ONLY') {
      acc.push({ kind: '/', type: 'ParameterNode' });
    }
    prev = p.kind;
    acc.push(p);
  }
  return (
    <code className="flex my-5 group signature">
      {node.kind.indexOf('async') !== -1 ||
      node.kind.indexOf('coroutine') !== -1
        ? 'async '
        : ''}
      def {node.target_name}(
      <>
        {/* TODO: insert the / from positional only */}
        {acc.map((parameter, index, array) => {
          if (index + 1 === array.length) {
            return <MyST ast={parameter} />;
          } else {
            return (
              <>
                <MyST ast={parameter} />
                {', '}
              </>
            );
          }
        })}
      </>
      {')'}
      <span className="ret-ann">
        {'->'} {node.return_annotation.data}
      </span>
      :
    </code>
  );
};

const Directive = ({ node }: { node: any }) => {
  const dom = node.domain !== null ? ':' + node.domain : '';
  const role = node.role !== null ? ':' + node.role + ':' : '';
  return (
    <>
      <code className="not-implemented">
        <span>
          {dom}
          {role}`{node.value}`
        </span>
      </code>
    </>
  );
};

const LOC = {
  signature: SignatureRenderer,
  Directive: Directive,
  DefList: DefList,
  Parameters: Parameters,
  ParameterNode: ParameterNodeRenderer,
  Param: Param,
  MUnimpl: MUnimpl,
  DefaultComponent: DefaultComponent,
  link: MyLink
};
export const RENDERERS = { ...DEFAULT_RENDERERS, ...LOC };

export function MyPapyri({ node }: { node: any }) {
  return <MyST ast={node.children} />;
}
