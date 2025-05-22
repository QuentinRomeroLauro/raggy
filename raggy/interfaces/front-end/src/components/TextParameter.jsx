import './TextParameter.css';

export default function TextParameter({ title, text, setText=() => {}}) {

    return (
        <div className="parameter">
            <span className="parameter-name">{title}</span>
            <input type="text" className="parameter-value" value={text} onChange={(event) => setText(event.target.value)} />
        </div>
    );
}
