const PillButton = ({ text, color, onClick }) => {
    const handleClick = (e) => {
        e.stopPropagation();
        onClick();
    };

    return (
        <button
            style={{
                backgroundColor: color ? color : "#B8DDFF",
                borderRadius: '15px',
                padding: '3px 8px',
                border: 'none',
                cursor: 'pointer',
                display: 'inline-block',
                width: 'fit-content',
                transition: 'transform 0.2s, background-color 0.2s',
                color: 'black',
                fontSize: '20px', // Increased text size
            }}
            onClick={(e) => {
                e.target.style.transform = 'scale(0.95)';
                setTimeout(() => {
                    e.target.style.transform = 'scale(1)';
                }, 150);
                handleClick(e);
            }}
            onMouseDown={(e) => (e.target.style.transform = 'scale(0.95)')}
            onMouseUp={(e) => (e.target.style.transform = 'scale(1)')}
            onMouseLeave={(e) => (e.target.style.transform = 'scale(1)')}
        >
            {text}
        </button>
    );
};

export default PillButton;
