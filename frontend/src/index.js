import React from "react";
import ReactDOM from "react-dom/client";
import "./index.css";
import { ThemeProvider } from "@myst-theme/providers";
import { MyST, DEFAULT_RENDERERS } from "myst-to-react";

import { fromMarkdown } from "mdast-util-from-markdown";

const Foo = ({ node }) => {
  if (!node.children) return <span>{node.value}</span>;
  return (
    <div className="not-implemented">
      <MyST ast={node.children} />
    </div>
  );
};

const MUnimpl = ({ node }) => {
  if (!node.children) return <span>{node.value}</span>;
  return (
    <div className="not-implemented unimpl">
      <MyST ast={node.children} />
    </div>
  );
};

const Param = ({ node }) => {
  return (
    <>
      <dt>
        {node.param}: {node.type_}
      </dt>
      <dd>
        {node.desc.map((sub) => (
          <MyST ast={sub} />
        ))}
      </dd>
    </>
  );
};
const Parameters = ({ node }) => {
  return (
    <dl>
      {node.children.map((item) => (
        <MyST ast={item} />
      ))}
    </dl>
  );
};

const DefList = ({ node }) => {
  return (
    <dl>
      {node.children.map((item) => (
        <>
          <dt>
            <MyST ast={item.dt} />
          </dt>
          <dd>
            {item.dd.map((sub) => (
              <MyST ast={sub} />
            ))}
          </dd>
        </>
      ))}
    </dl>
  );
};

// render a single parameter in a signature
const ParameterNodeRenderer = ({ node }) => {
  if (node.kind === "/") {
    return <span>/</span>;
  }
  let comp = [];
  let name = "";
  if (node.kind === "VAR_POSITIONAL") {
    name = "*";
  }
  if (node.kind === "VAR_KEYWORD") {
    name += "**";
  }
  name += node.name;

  comp = [<span>{name}</span>];

  if (node.annotation.type !== "Empty") {
    comp.push(<span className="type-ann">{": " + node.annotation.data}</span>);
  }
  if (node.default.type !== "Empty") {
    comp.push(<span className="default-value">{"=" + node.default.data}</span>);
  }

  return <>{comp}</>;
};

const SignatureRenderer = ({ node }) => {
  let prev = "";
  let acc = [];
  for (let i = 0; i < node.parameters.length; i++) {
    const p = node.parameters[i];
    if (p.kind !== "POSITIONAL_ONLY" && prev === "POSITIONAL_ONLY") {
      acc.push({ kind: "/", type: "ParameterNode" });
    }
    prev = p.kind;
    acc.push(p);
  }
  return (
    <code className="flex my-5 group signature">
      {node.kind.indexOf("async") !== -1 ||
      node.kind.indexOf("coroutine") !== -1
        ? "async "
        : ""}
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
                {", "}
              </>
            );
          }
        })}
      </>
      {")"}
      <span className="ret-ann">
        {"->"} {node.return_annotation.data}
      </span>
      :
    </code>
  );
};

const Directive = ({ node }) => {
  const dom = node.domain !== null ? ":" + node.domain : "";
  const role = node.role !== null ? ":" + node.role + ":" : "";
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
  DefaultComponent: Foo,
};
const RENDERERS = { ...DEFAULT_RENDERERS, ...LOC };

function MyComponent({ node }) {
  return <MyST ast={node.children} />;
}

//const tree = fromMarkdown("Some *emphasis*, **strong**, and `code`.");
//const mytree = {
//  type: "admonition",
//  children: [
//    { type: "text", value: "myValue" },
//    {
//      type: "signature",
//      value: "Foo",
//      children: [{ type: "text", value: "Child" }],
//    },
//  ],
//};

console.log("Loading X");

const render = (id, tree) => {
  const root = ReactDOM.createRoot(document.getElementById(id));
  root.render(
    <React.StrictMode>
      <ThemeProvider renderers={RENDERERS}>
        <MyComponent node={tree} />
      </ThemeProvider>
    </React.StrictMode>,
  );
};

window.render = render;
window.fromMarkdown = fromMarkdown;
