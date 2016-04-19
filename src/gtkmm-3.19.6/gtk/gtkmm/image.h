// Generated by gmmproc 2.47.4 -- DO NOT MODIFY!
#ifndef _GTKMM_IMAGE_H
#define _GTKMM_IMAGE_H


#include <glibmm/ustring.h>
#include <sigc++/sigc++.h>

/*
 * Copyright (C) 1998-2002 The gtkmm Development Team
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 */

#include <gtkmm/misc.h>
#include <gtkmm/iconset.h>
#include <gdkmm/pixbufanimation.h>
#include <giomm/icon.h>


#ifndef DOXYGEN_SHOULD_SKIP_THIS
typedef struct _GtkImage GtkImage;
typedef struct _GtkImageClass GtkImageClass;
#endif /* DOXYGEN_SHOULD_SKIP_THIS */


#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace Gtk
{ class Image_Class; } // namespace Gtk
#endif //DOXYGEN_SHOULD_SKIP_THIS

namespace Gtk
{


/** @addtogroup gtkmmEnums gtkmm Enums and Flags */

/** 
 *  @var ImageType IMAGE_EMPTY
 * There is no image displayed by the widget.
 * 
 *  @var ImageType IMAGE_PIXBUF
 * The widget contains a Gdk::Pixbuf.
 * 
 *  @var ImageType IMAGE_STOCK
 * The widget contains a [stock item name][gtkstock].
 * 
 *  @var ImageType IMAGE_ICON_SET
 * The widget contains a Gtk::IconSet.
 * 
 *  @var ImageType IMAGE_ANIMATION
 * The widget contains a Gdk::PixbufAnimation.
 * 
 *  @var ImageType IMAGE_ICON_NAME
 * The widget contains a named icon.
 * This image type was added in GTK+ 2.6.
 * 
 *  @var ImageType IMAGE_GICON
 * The widget contains a Icon.
 * This image type was added in GTK+ 2.14.
 * 
 *  @var ImageType IMAGE_SURFACE
 * The widget contains a #cairo_surface_t.
 * This image type was added in GTK+ 3.10.
 * 
 *  @enum ImageType
 * 
 * Describes the image data representation used by a Gtk::Image. If you
 * want to get the image from the widget, you can only get the
 * currently-stored representation. e.g.  if the
 * Gtk::Image::get_storage_type() returns Gtk::IMAGE_PIXBUF, then you can
 * call Gtk::Image::get_pixbuf() but not Gtk::Image::get_stock().  For empty
 * images, you can request any storage type (call any of the "get"
 * functions), but they will all return <tt>nullptr</tt> values.
 *
 * @ingroup gtkmmEnums
 */
enum ImageType
{
  IMAGE_EMPTY,
  IMAGE_PIXBUF,
  IMAGE_STOCK,
  IMAGE_ICON_SET,
  IMAGE_ANIMATION,
  IMAGE_ICON_NAME,
  IMAGE_GICON,
  IMAGE_SURFACE
};

} // namespace Gtk


#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace Glib
{

template <>
class Value<Gtk::ImageType> : public Glib::Value_Enum<Gtk::ImageType>
{
public:
  static GType value_type() G_GNUC_CONST;
};

} // namespace Glib
#endif /* DOXYGEN_SHOULD_SKIP_THIS */


namespace Gtk
{


/** A widget displaying an image.
 *
 * The Gtk::Image widget displays an image. Various kinds of object can be
 * displayed as an image; most typically, you would load a Gdk::Pixbuf ("pixel
 * buffer") from a file, and then display that.
 *
 * Gtk::Image is a subclass of Gtk::Misc, which implies that you can align it
 * (center, left, right) and add padding to it, using Gtk::Misc methods.
 *
 * Gtk::Image is a "no window" widget (has no Gdk::Window of its own), so by
 * default does not receive events. If you want to receive events on the
 * image, such as button clicks, place the image inside a Gtk::EventBox, then
 * connect to the event signals on the event box.
 *
 * The Image widget looks like this:
 * @image html image1.png
 *
 * @ingroup Widgets
 */

class Image : public Misc
{
  public:
#ifndef DOXYGEN_SHOULD_SKIP_THIS
  typedef Image CppObjectType;
  typedef Image_Class CppClassType;
  typedef GtkImage BaseObjectType;
  typedef GtkImageClass BaseClassType;
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

  Image(Image&& src) noexcept;
  Image& operator=(Image&& src) noexcept;

  // noncopyable
  Image(const Image&) = delete;
  Image& operator=(const Image&) = delete;

  ~Image() noexcept override;

#ifndef DOXYGEN_SHOULD_SKIP_THIS

private:
  friend class Image_Class;
  static CppClassType image_class_;

protected:
  explicit Image(const Glib::ConstructParams& construct_params);
  explicit Image(GtkImage* castitem);

#endif /* DOXYGEN_SHOULD_SKIP_THIS */

public:

  /** Get the GType for this class, for use with the underlying GObject type system.
   */
  static GType get_type()      G_GNUC_CONST;

#ifndef DOXYGEN_SHOULD_SKIP_THIS


  static GType get_base_type() G_GNUC_CONST;
#endif

  ///Provides access to the underlying C GtkObject.
  GtkImage*       gobj()       { return reinterpret_cast<GtkImage*>(gobject_); }

  ///Provides access to the underlying C GtkObject.
  const GtkImage* gobj() const { return reinterpret_cast<GtkImage*>(gobject_); }


public:
  //C++ methods used to invoke GTK+ virtual functions:

protected:
  //GTK+ Virtual Functions (override these to change behaviour):

  //Default Signal Handlers::


private:

  
public:

  Image();

  /** Creates an Image widget displaying the file @a filename.
   * If the file isn't found or can't be loaded, the resulting Gtk::Image will display a "broken image" icon.
   *
   * If the file contains an animation, the image will contain an animation.
   *
   * If you need to detect failures to load the file, use Gdk::Pixbuf::create_from_file() to load the file yourself,
   * then create the GtkImage from the pixbuf. (Or for animations, use Gdk::PixbufAnimation::create_from_file()).
   *
   * The storage type (get_storage_type()) of the returned image is not defined. It will be whatever is appropriate for displaying the file.
   */
    explicit Image(const std::string& file);


  /** Creates a new Image widget displaying @a pixbuf.
   * Note that this just creates an GtkImage from the pixbuf. The Gtk::Image created will not react to state changes.
   * Should you want that, you should use the default constructor and set_from_icon_name().
   */
    explicit Image(const Glib::RefPtr<Gdk::Pixbuf>& pixbuf);

   // deprecated since 3.10

  //We don't wrap gtk_image_new_from_icon_name() to avoid a clash with the from-filename constructor.
  //But we do wrap gtk_image_set_from_icon_name()
  

#ifndef GTKMM_DISABLE_DEPRECATED
  /** Creates am Image displaying a stock icon.
   * Sample stock icon identifiers are Gtk::Stock::OPEN, Gtk::Stock::EXIT. Sample stock sizes are Gtk::ICON_SIZE_MENU, Gtk::ICON_SIZE_SMALL_TOOLBAR.
   * If the stock icon name isn't known, a "broken image" icon will be displayed instead.
   * You can register your own stock icon names - see Gtk::IconFactory::add().
   * @param stock_id A stock icon.
   * @param size A stock icon size.
   * @deprecated Use the default constructor and set_from_icon_name() instead.
   */
  Image(const Gtk::StockID& stock_id, IconSize size);
#endif //GTKMM_DISABLE_DEPRECATED

#ifndef GTKMM_DISABLE_DEPRECATED
  /** @deprecated Use the default constructor and set_from_icon_name() instead.
   */
  Image(const Glib::RefPtr<IconSet>& icon_set, IconSize size);
#endif //GTKMM_DISABLE_DEPRECATED

  explicit Image(const Glib::RefPtr<Gdk::PixbufAnimation>& animation);

  
  /** See the Image::Image(const std::string& file) constructor for details.
   * 
   * @param filename A filename.
   */
  void set(const std::string& filename);
  
  /** See new_from_resource() for details.
   * 
   * @param resource_path A resource path or <tt>nullptr</tt>.
   */
  void set_from_resource(const std::string& resource_path);
  
  /** See the Image::Image(const Glib::RefPtr<Gdk::Pixbuf>& pixbuf) constructor for details.
   * 
   * @param pixbuf A #Gdk::Pixbuf.
   */
  void set(const Glib::RefPtr<Gdk::Pixbuf>& pixbuf);
  
#ifndef GTKMM_DISABLE_DEPRECATED

  /** See the Image::Image(const Gtk::StockID& stock_id, IconSize size) constructor for details.
   * 
   * @deprecated Use set_from_icon_name() instead.
   * 
   * @param stock_id A stock icon name.
   * @param size A stock icon size.
   */
  void set(const Gtk::StockID& stock_id, IconSize size);
#endif // GTKMM_DISABLE_DEPRECATED


#ifndef GTKMM_DISABLE_DEPRECATED

  /** See new_from_icon_set() for details.
   * 
   * Deprecated: 3.10: Use set_from_icon_name() instead.
   * 
   * @deprecated Use set_from_icon_name() instead.
   * 
   * @param icon_set A Gtk::IconSet.
   * @param size A stock icon size.
   */
  void set(const Glib::RefPtr<IconSet>& icon_set, IconSize size);
#endif // GTKMM_DISABLE_DEPRECATED


  /** Causes the Gtk::Image to display the given animation (or display
   * nothing, if you set the animation to <tt>nullptr</tt>).
   * 
   * @param animation The Gdk::PixbufAnimation.
   */
  void set(const Glib::RefPtr<Gdk::PixbufAnimation>& animation);
  
  /** See new_from_gicon() for details.
   * 
   * @newin{2,14}
   * 
   * @param icon An icon.
   * @param size An icon size.
   */
  void set(const Glib::RefPtr<const Gio::Icon>& icon, IconSize size);

 
  /** See new_from_surface() for details.
   * 
   * @newin{3,10}
   * 
   * @param surface A cairo_surface_t.
   */
  void set(const ::Cairo::RefPtr< ::Cairo::Surface>& surface);

  
  /** Causes the Image to display an icon from the current icon theme.
   * If the icon name isn't known, a "broken image" icon will be
   * displayed instead.  If the current icon theme is changed, the icon
   * will be updated appropriately.
   * 
   * @newin{2,6}
   * 
   * @param icon_name An icon name.
   * @param size An icon size.
   */
  void set_from_icon_name(const Glib::ustring& icon_name, IconSize size);


  /** Resets the image to be empty.
   * 
   * @newin{2,8}
   */
  void clear();

  
  /** Gets the type of representation being used by the Gtk::Image
   * to store image data. If the Gtk::Image has no image data,
   * the return value will be Gtk::IMAGE_EMPTY.
   * 
   * @return Image representation being used.
   */
  ImageType get_storage_type() const;

  
  /** Gets the Gdk::Pixbuf being displayed by the Gtk::Image.
   * The storage type of the image must be Gtk::IMAGE_EMPTY or
   * Gtk::IMAGE_PIXBUF (see get_storage_type()).
   * The caller of this function does not own a reference to the
   * returned pixbuf.
   * 
   * @return The displayed pixbuf, or <tt>nullptr</tt> if
   * the image is empty.
   */
  Glib::RefPtr<Gdk::Pixbuf> get_pixbuf();
  
  /** Gets the Gdk::Pixbuf being displayed by the Gtk::Image.
   * The storage type of the image must be Gtk::IMAGE_EMPTY or
   * Gtk::IMAGE_PIXBUF (see get_storage_type()).
   * The caller of this function does not own a reference to the
   * returned pixbuf.
   * 
   * @return The displayed pixbuf, or <tt>nullptr</tt> if
   * the image is empty.
   */
  Glib::RefPtr<const Gdk::Pixbuf> get_pixbuf() const;

#ifndef GTKMM_DISABLE_DEPRECATED
  /** @deprecated Use get_icon_name() instead.
   */
  void get_stock(Gtk::StockID& stock_id, IconSize& size) const;
#endif //GTKMM_DISABLE_DEPRECATED

#ifndef GTKMM_DISABLE_DEPRECATED
  /** @deprecated Use get_icon_name() instead.
   */
  void get_icon_set(Glib::RefPtr<IconSet>& icon_set, IconSize& size) const;
#endif //GTKMM_DISABLE_DEPRECATED

  
  /** Gets the Gdk::PixbufAnimation being displayed by the Gtk::Image.
   * The storage type of the image must be Gtk::IMAGE_EMPTY or
   * Gtk::IMAGE_ANIMATION (see get_storage_type()).
   * The caller of this function does not own a reference to the
   * returned animation.
   * 
   * @return The displayed animation, or <tt>nullptr</tt> if
   * the image is empty.
   */
  Glib::RefPtr<Gdk::PixbufAnimation> get_animation();
  
  /** Gets the Gdk::PixbufAnimation being displayed by the Gtk::Image.
   * The storage type of the image must be Gtk::IMAGE_EMPTY or
   * Gtk::IMAGE_ANIMATION (see get_storage_type()).
   * The caller of this function does not own a reference to the
   * returned animation.
   * 
   * @return The displayed animation, or <tt>nullptr</tt> if
   * the image is empty.
   */
  Glib::RefPtr<const Gdk::PixbufAnimation> get_animation() const;

 /** Gets the Gio::Icon and size being displayed by the Gtk::Image.
  * The storage type of the image must be IMAGE_EMPTY or
  * IMAGE_GICON (see get_storage_type()).
  *
  * @param icon_size A place to store an icon size.
  *
  * @newin{2,14}
  */
  Glib::RefPtr<Gio::Icon> get_gicon(Gtk::IconSize& icon_size);

 /** Gets the Gio::Icon and size being displayed by the Gtk::Image.
  * The storage type of the image must be IMAGE_EMPTY or
  * IMAGE_GICON (see get_storage_type()).
  *
  * @param icon_size A place to store an icon size.
  *
  * @newin{2,14}
  */
  Glib::RefPtr<const Gio::Icon> get_gicon(Gtk::IconSize& icon_size) const;
  

  Glib::ustring get_icon_name() const;
  Glib::ustring get_icon_name(IconSize& size);
  

  /** Gets the pixel size used for named icons.
   * 
   * @newin{2,6}
   * 
   * @return The pixel size used for named icons.
   */
  int get_pixel_size() const;
 
  /** Sets the pixel size to use for named icons. If the pixel size is set
   * to a value != -1, it is used instead of the icon size set by
   * set_from_icon_name().
   * 
   * @newin{2,6}
   * 
   * @param pixel_size The new pixel size.
   */
  void set_pixel_size(int pixel_size);

  /** A GdkPixbuf to display.
   *
   * @return A PropertyProxy that allows you to get or set the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy< Glib::RefPtr<Gdk::Pixbuf> > property_pixbuf() ;

/** A GdkPixbuf to display.
   *
   * @return A PropertyProxy_ReadOnly that allows you to get the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy_ReadOnly< Glib::RefPtr<Gdk::Pixbuf> > property_pixbuf() const;

  /** Filename to load and display.
   *
   * @return A PropertyProxy that allows you to get or set the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy< Glib::ustring > property_file() ;

/** Filename to load and display.
   *
   * @return A PropertyProxy_ReadOnly that allows you to get the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy_ReadOnly< Glib::ustring > property_file() const;

  
#ifndef GTKMM_DISABLE_DEPRECATED

/** Stock ID for a stock image to display.
   *
   * Deprecated: 3.10: Use Gtk::Image::property_icon_name() instead.
   * 
   * @deprecated Use property_icon_name() instead.
   *
   * @return A PropertyProxy that allows you to get or set the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy< Glib::ustring > property_stock() ;

/** Stock ID for a stock image to display.
   *
   * Deprecated: 3.10: Use Gtk::Image::property_icon_name() instead.
   * 
   * @deprecated Use property_icon_name() instead.
   *
   * @return A PropertyProxy_ReadOnly that allows you to get the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy_ReadOnly< Glib::ustring > property_stock() const;

#endif // GTKMM_DISABLE_DEPRECATED

  
#ifndef GTKMM_DISABLE_DEPRECATED

/** Icon set to display.
   *
   * Deprecated: 3.10: Use Gtk::Image::property_icon_name() instead.
   * 
   * @deprecated Use property_icon_name() instead.
   *
   * @return A PropertyProxy that allows you to get or set the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy< Glib::RefPtr<IconSet> > property_icon_set() ;

/** Icon set to display.
   *
   * Deprecated: 3.10: Use Gtk::Image::property_icon_name() instead.
   * 
   * @deprecated Use property_icon_name() instead.
   *
   * @return A PropertyProxy_ReadOnly that allows you to get the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy_ReadOnly< Glib::RefPtr<IconSet> > property_icon_set() const;

#endif // GTKMM_DISABLE_DEPRECATED

  /** Symbolic size to use for stock icon, icon set or named icon.
   *
   * @return A PropertyProxy that allows you to get or set the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy< int > property_icon_size() ;

/** Symbolic size to use for stock icon, icon set or named icon.
   *
   * @return A PropertyProxy_ReadOnly that allows you to get the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy_ReadOnly< int > property_icon_size() const;

  /** The "pixel-size" property can be used to specify a fixed size
   * overriding the Gtk::Image::property_icon_size() property for images of type
   * Gtk::IMAGE_ICON_NAME.
   * 
   * @newin{2,6}
   *
   * @return A PropertyProxy that allows you to get or set the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy< int > property_pixel_size() ;

/** The "pixel-size" property can be used to specify a fixed size
   * overriding the Gtk::Image::property_icon_size() property for images of type
   * Gtk::IMAGE_ICON_NAME.
   * 
   * @newin{2,6}
   *
   * @return A PropertyProxy_ReadOnly that allows you to get the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy_ReadOnly< int > property_pixel_size() const;

  /** GdkPixbufAnimation to display.
   *
   * @return A PropertyProxy that allows you to get or set the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy< Glib::RefPtr<Gdk::PixbufAnimation> > property_pixbuf_animation() ;

/** GdkPixbufAnimation to display.
   *
   * @return A PropertyProxy_ReadOnly that allows you to get the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy_ReadOnly< Glib::RefPtr<Gdk::PixbufAnimation> > property_pixbuf_animation() const;

  /** The name of the icon in the icon theme. If the icon theme is
   * changed, the image will be updated automatically.
   * 
   * @newin{2,6}
   *
   * @return A PropertyProxy that allows you to get or set the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy< Glib::ustring > property_icon_name() ;

/** The name of the icon in the icon theme. If the icon theme is
   * changed, the image will be updated automatically.
   * 
   * @newin{2,6}
   *
   * @return A PropertyProxy_ReadOnly that allows you to get the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy_ReadOnly< Glib::ustring > property_icon_name() const;

  /** The representation being used for image data.
   *
   * @return A PropertyProxy_ReadOnly that allows you to get the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy_ReadOnly< ImageType > property_storage_type() const;


  /** The GIcon displayed in the GtkImage. For themed icons,
   * If the icon theme is changed, the image will be updated
   * automatically.
   * 
   * @newin{2,14}
   *
   * @return A PropertyProxy that allows you to get or set the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy< Glib::RefPtr<Gio::Icon> > property_gicon() ;

/** The GIcon displayed in the GtkImage. For themed icons,
   * If the icon theme is changed, the image will be updated
   * automatically.
   * 
   * @newin{2,14}
   *
   * @return A PropertyProxy_ReadOnly that allows you to get the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy_ReadOnly< Glib::RefPtr<Gio::Icon> > property_gicon() const;

  /** Whether the icon displayed in the GtkImage will use
   * standard icon names fallback. The value of this property
   * is only relevant for images of type Gtk::IMAGE_ICON_NAME
   * and Gtk::IMAGE_GICON.
   * 
   * @newin{3,0}
   *
   * @return A PropertyProxy that allows you to get or set the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy< bool > property_use_fallback() ;

/** Whether the icon displayed in the GtkImage will use
   * standard icon names fallback. The value of this property
   * is only relevant for images of type Gtk::IMAGE_ICON_NAME
   * and Gtk::IMAGE_GICON.
   * 
   * @newin{3,0}
   *
   * @return A PropertyProxy_ReadOnly that allows you to get the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy_ReadOnly< bool > property_use_fallback() const;

  /** A path to a resource file to display.
   * 
   * @newin{3,8}
   *
   * @return A PropertyProxy that allows you to get or set the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy< std::string > property_resource() ;

/** A path to a resource file to display.
   * 
   * @newin{3,8}
   *
   * @return A PropertyProxy_ReadOnly that allows you to get the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy_ReadOnly< std::string > property_resource() const;

  /** A cairo_surface_t to display.
   *
   * @return A PropertyProxy that allows you to get or set the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy< ::Cairo::RefPtr< ::Cairo::Surface> > property_surface() ;

/** A cairo_surface_t to display.
   *
   * @return A PropertyProxy_ReadOnly that allows you to get the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy_ReadOnly< ::Cairo::RefPtr< ::Cairo::Surface> > property_surface() const;


};

} //namespace Gtk


namespace Glib
{
  /** A Glib::wrap() method for this object.
   * 
   * @param object The C instance.
   * @param take_copy False if the result should take ownership of the C instance. True if it should take a new copy or ref.
   * @result A C++ instance that wraps this C instance.
   *
   * @relates Gtk::Image
   */
  Gtk::Image* wrap(GtkImage* object, bool take_copy = false);
} //namespace Glib


#endif /* _GTKMM_IMAGE_H */

