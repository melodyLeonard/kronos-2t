import UIKit

/// Apple-style emoji keyboard: horizontally scrolling grid, category bar, ABC and delete keys.
final class EmojiKeyboard: UIView, UICollectionViewDataSource, UICollectionViewDelegate {
    var onEmoji: ((String) -> Void)?
    var onABC: (() -> Void)?
    var onBackspace: (() -> Void)?

    private let dark: Bool
    private var sections: [EmojiCatalog.Category]
    private let grid: UICollectionView
    private let catBar = UIStackView()
    private var catButtons: [UIButton] = []
    private let cell = 34.0
    private let title = UILabel()

    init(dark: Bool) {
        self.dark = dark
        let recent = Prefs.recentEmoji
        var s = EmojiCatalog.categories
        if !recent.isEmpty { s.insert(.init(name: "Frequently Used", symbol: "clock", items: recent), at: 0) }
        sections = s
        let layout = UICollectionViewFlowLayout()
        layout.scrollDirection = .horizontal
        layout.itemSize = CGSize(width: cell, height: cell)
        layout.minimumInteritemSpacing = 2
        layout.minimumLineSpacing = 4
        layout.sectionInset = .init(top: 2, left: 8, bottom: 2, right: 14)
        grid = UICollectionView(frame: .zero, collectionViewLayout: layout)
        super.init(frame: .zero)
        backgroundColor = dark ? UIColor(white: 0.17, alpha: 1) : UIColor(red: 0.82, green: 0.827, blue: 0.851, alpha: 1)
        grid.backgroundColor = .clear
        grid.showsHorizontalScrollIndicator = false
        grid.disableEdgeBlur()
        grid.register(EmojiCell.self, forCellWithReuseIdentifier: "e")
        grid.dataSource = self
        grid.delegate = self
        grid.translatesAutoresizingMaskIntoConstraints = false
        addSubview(grid)
        title.font = .systemFont(ofSize: 12, weight: .semibold)
        title.textColor = dark ? UIColor(white: 0.7, alpha: 1) : UIColor(white: 0.35, alpha: 1)
        title.translatesAutoresizingMaskIntoConstraints = false
        addSubview(title)

        let ink: UIColor = dark ? .white : .black
        let abc = UIButton(type: .system)
        abc.setTitle("ABC", for: .normal)
        abc.titleLabel?.font = .systemFont(ofSize: 16)
        abc.tintColor = ink
        abc.addAction(UIAction { [weak self] _ in self?.onABC?() }, for: .touchUpInside)
        let del = UIButton(type: .system)
        del.setImage(UIImage(systemName: "delete.left", withConfiguration: UIImage.SymbolConfiguration(pointSize: 18)), for: .normal)
        del.tintColor = ink
        del.addAction(UIAction { [weak self] _ in self?.onBackspace?() }, for: .touchUpInside)

        catBar.axis = .horizontal
        catBar.distribution = .fillEqually
        for (i, c) in sections.enumerated() {
            let b = UIButton(type: .system)
            b.setImage(UIImage(systemName: c.symbol, withConfiguration: UIImage.SymbolConfiguration(pointSize: 15)), for: .normal)
            b.tintColor = dark ? UIColor(white: 0.7, alpha: 1) : UIColor(white: 0.35, alpha: 1)
            b.layer.cornerRadius = 14
            b.addAction(UIAction { [weak self] _ in self?.jump(to: i) }, for: .touchUpInside)
            catButtons.append(b)
            catBar.addArrangedSubview(b)
        }
        let bar = UIStackView(arrangedSubviews: [abc, catBar, del])
        bar.axis = .horizontal
        bar.spacing = 8
        bar.translatesAutoresizingMaskIntoConstraints = false
        addSubview(bar)
        NSLayoutConstraint.activate([
            title.topAnchor.constraint(equalTo: topAnchor, constant: 14),
            title.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 12),
            grid.topAnchor.constraint(equalTo: topAnchor, constant: 36),
            grid.leadingAnchor.constraint(equalTo: leadingAnchor),
            grid.trailingAnchor.constraint(equalTo: trailingAnchor),
            grid.heightAnchor.constraint(equalToConstant: 5 * cell + 4 * 4 + 4),
            bar.topAnchor.constraint(equalTo: grid.bottomAnchor, constant: 2),
            bar.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 12),
            bar.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -12),
            bar.heightAnchor.constraint(equalToConstant: 28),
            abc.widthAnchor.constraint(equalToConstant: 44),
            del.widthAnchor.constraint(equalToConstant: 36),
        ])
        highlight(0)
    }
    required init?(coder: NSCoder) { fatalError() }

    private func jump(to section: Int) {
        guard !sections[section].items.isEmpty else { return }
        grid.scrollToItem(at: IndexPath(item: 0, section: section), at: .left, animated: true)
        highlight(section)
    }

    private func highlight(_ section: Int) {
        title.text = sections[section].name.uppercased()
        for (i, b) in catButtons.enumerated() {
            b.backgroundColor = i == section ? (dark ? UIColor(white: 0.35, alpha: 1) : UIColor(white: 1, alpha: 0.7)) : .clear
            b.tintColor = i == section ? (dark ? .white : .black) : (dark ? UIColor(white: 0.7, alpha: 1) : UIColor(white: 0.35, alpha: 1))
        }
    }

    func scrollViewDidScroll(_ scrollView: UIScrollView) {
        let p = CGPoint(x: scrollView.contentOffset.x + 20, y: 20)
        if let ip = grid.indexPathForItem(at: p) { highlight(ip.section) }
    }

    func numberOfSections(in collectionView: UICollectionView) -> Int { sections.count }
    func collectionView(_ collectionView: UICollectionView, numberOfItemsInSection section: Int) -> Int { sections[section].items.count }
    func collectionView(_ collectionView: UICollectionView, cellForItemAt indexPath: IndexPath) -> UICollectionViewCell {
        let c = collectionView.dequeueReusableCell(withReuseIdentifier: "e", for: indexPath) as! EmojiCell
        c.label.text = sections[indexPath.section].items[indexPath.item]
        return c
    }
    func collectionView(_ collectionView: UICollectionView, didSelectItemAt indexPath: IndexPath) {
        let e = sections[indexPath.section].items[indexPath.item]
        Prefs.touchEmoji(e)
        onEmoji?(e)
    }
}

extension UIScrollView {
    /// iOS 26 blurs content under scroll edges by default; a keyboard has no bar to scroll under.
    func disableEdgeBlur() {
        if #available(iOS 26.0, *) {
            topEdgeEffect.isHidden = true; bottomEdgeEffect.isHidden = true
            leftEdgeEffect.isHidden = true; rightEdgeEffect.isHidden = true
        }
    }
}

final class EmojiCell: UICollectionViewCell {
    let label = UILabel()
    override init(frame: CGRect) {
        super.init(frame: frame)
        label.font = .systemFont(ofSize: 27)
        label.textAlignment = .center
        label.frame = bounds
        label.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        contentView.addSubview(label)
    }
    required init?(coder: NSCoder) { fatalError() }
    override var isHighlighted: Bool {
        didSet { contentView.backgroundColor = isHighlighted ? UIColor(white: 0.5, alpha: 0.3) : .clear; contentView.layer.cornerRadius = 8 }
    }
}
